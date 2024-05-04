import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts import MCTSExperience
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
        return assign_target_values(agent.trajectory, board)

    def assign_target_values(traj: list[MCTSExperience], board: Board) -> list[MCTSExperience]:
        last_exp = traj[-1]
        if board.info.game_over:
            # -1 on the reward, since that state is not on the trajectory
            value = -float(board.info.reward)
        else:
            value = heuristic_fcn(last_exp.state) if heuristic else traj[-1].v_target

        last_exp.v_target = value
        if reassign_values:
            for experience in reversed(traj):
                experience.v_target = value
                value *= -1.0
        return traj

    def initialize_agent() -> None:
        while not db.first_weights_published():
            time.sleep(0.1)
        update_weights()

    def update_weights() -> int:
        if db.weights_is_latest(current_weights):
            return current_weights
        weight_index, weights = db.fetch_latest_weights_with_index()
        agent.nn.load_state_dict(weights)
        logger.info(f"updated weights {current_weights}->{weight_index}")
        return weight_index

    init_rootlogger(verbose=verbose)
    agent = MCTSAgent(
        size,
        n_rollouts,
        rollout_depth,
    )
    db = TrainingDB(host=Environmnet.host_redis)
    heuristic_fcn = Heuristic(size, subtract_opponent=True)

    current_weights = 0
    initialize_agent()
    while True:
        traj = rollout_trajectory()
        db.post_trajectory(traj)
        logger.info(f"worker posts {len(traj)} samples")
        current_weights = update_weights()
