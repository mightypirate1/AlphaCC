import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    DefaultAssignmentStrategyWithHeuristic,
)
from alpha_cc.config import Environmnet
from alpha_cc.db import TrainingDB
from alpha_cc.engine import Board
from alpha_cc.entrypoints.logs import init_rootlogger
from alpha_cc.runtimes import TrainingRunTime

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--max-game-length", type=int, default=500)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int,
    heuristic: bool,
    verbose: bool,
) -> None:
    def initialize_agent() -> int:
        while not db.first_weights_published():
            time.sleep(0.1)
        return update_weights()

    def update_weights() -> int:
        if db.weights_is_latest(current_weights):
            return current_weights
        weight_index, weights = db.fetch_latest_weights_with_index()
        agent.nn.load_state_dict(weights)
        logger.info(f"updated weights {current_weights}->{weight_index}")
        return weight_index

    init_rootlogger(verbose=verbose)
    value_assignment_strategy = (
        DefaultAssignmentStrategyWithHeuristic(size) if heuristic else DefaultAssignmentStrategy()
    )
    agent = MCTSAgent(
        size,
        n_rollouts,
        rollout_depth,
    )
    training_runtime = TrainingRunTime(
        Board(size),
        agent,
        value_assignment_strategy=value_assignment_strategy,
    )
    db = TrainingDB(host=Environmnet.host_redis)

    # the trainer needs to start and flush the db, so we wait
    time.sleep(10)  # TODO: figure out why workers can start before trainer
    current_weights = 0
    current_weights = initialize_agent()
    while True:
        traj = training_runtime.play_game(max_game_length=max_game_length)
        db.post_trajectory(traj)
        logger.debug(f"worker posts {len(traj)} samples")
        current_weights = update_weights()
