import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.agents.value_assignment import DefaultAssignmentStrategy
from alpha_cc.config import Environmnet
from alpha_cc.db import TrainingDB
from alpha_cc.engine import Board
from alpha_cc.entrypoints.logs import init_rootlogger

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--max-game-length", type=int, default=500)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--reassign-values", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int,
    heuristic: bool,
    reassign_values: bool,
    verbose: bool,
) -> None:
    def rollout_trajectory() -> list[MCTSExperience]:
        board = Board(size)
        agent.on_game_start()
        agent.nn.eval()
        while not board.info.game_over and board.info.duration < max_game_length:
            moves = board.get_moves()
            a = agent.choose_move(board, training=True)
            board = board.apply(moves[a])
        agent.on_game_end(board)
        return agent.trajectory

    def initialize_agent() -> None:
        while not db.first_weights_published():
            time.sleep(0.1)
        update_weights()

    def update_weights() -> int:
        if db.weights_is_latest(current_weights):
            return current_weights
        weight_index, weights = db.fetch_latest_weights_with_index()
        agent.nn.load_state_dict(weights)
        logger.debug(f"updated weights {current_weights}->{weight_index}")
        return weight_index

    init_rootlogger(verbose=verbose)
    value_assignment_strategy = DefaultAssignmentStrategy(gamma=0.99) if reassign_values else None
    agent = MCTSAgent(
        size,
        n_rollouts,
        rollout_depth,
        apply_heuristic=heuristic,
        value_assignment_strategy=value_assignment_strategy,
    )
    db = TrainingDB(host=Environmnet.host_redis)

    current_weights = 0
    initialize_agent()
    while True:
        traj = rollout_trajectory()
        db.post_trajectory(traj)
        logger.debug(f"worker posts {len(traj)} samples")
        current_weights = update_weights()
