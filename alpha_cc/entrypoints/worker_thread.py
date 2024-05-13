import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    DefaultAssignmentStrategyWithHeuristic,
)
from alpha_cc.config import Environment
from alpha_cc.db import TrainingDB
from alpha_cc.engine import Board
from alpha_cc.entrypoints.logs import init_rootlogger
from alpha_cc.runtimes import TrainingRunTime

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--max-game-length", type=int)
@click.option("--dirichlet-noise-weight", type=float, default=0.0)
@click.option("--argmax-delay", type=int, default=None)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int | None,
    dirichlet_noise_weight: float,
    argmax_delay: int | None,
    heuristic: bool,
    verbose: bool,
) -> None:
    init_rootlogger(verbose=verbose)
    value_assignment_strategy = (
        DefaultAssignmentStrategyWithHeuristic(size) if heuristic else DefaultAssignmentStrategy()
    )
    agent = MCTSAgent(
        Environment.host_redis,
        n_rollouts,
        rollout_depth,
        dirichlet_weight=dirichlet_noise_weight,
        argmax_delay=argmax_delay,
    )
    training_runtime = TrainingRunTime(
        Board(size),
        agent,
        value_assignment_strategy=value_assignment_strategy,
    )
    db = TrainingDB(host=Environment.host_redis)

    # the trainer needs to start and flush the db, so we wait
    time.sleep(10)  # TODO: figure out why workers can start before trainer
    while True:
        traj = training_runtime.play_game(max_game_length=max_game_length)
        if max_game_length is not None and len(traj) < max_game_length:
            db.post_trajectory(traj)
            logger.debug(f"worker posts {len(traj)} samples")
        else:
            logger.warn(f"worker dropped {len(traj)} samples")
