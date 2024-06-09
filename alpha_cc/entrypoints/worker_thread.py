import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    HeuristicAssignmentStrategy,
)
from alpha_cc.config import Environment
from alpha_cc.db import TrainingDB
from alpha_cc.engine import Board
from alpha_cc.logs import init_rootlogger
from alpha_cc.runtimes import TournamentRuntime, TrainingRunTime

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--max-game-length", type=int)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--dirichlet-noise-weight", type=float, default=0.0)
@click.option("--argmax-delay", type=int, default=None)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--gamma", type=float, default=1.0)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int | None,
    rollout_gamma: float,
    dirichlet_noise_weight: float,
    argmax_delay: int | None,
    heuristic: bool,
    gamma: float,
    verbose: bool,
) -> None:
    def create_model(channel: int) -> MCTSAgent:
        return MCTSAgent(
            Environment.host_redis,
            pred_channel=channel,
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            rollout_gamma=rollout_gamma,
            dirichlet_weight=dirichlet_noise_weight,
            argmax_delay=argmax_delay,
        )

    init_rootlogger(verbose=verbose)
    value_assignment_strategy = (
        HeuristicAssignmentStrategy(size, gamma) if heuristic else DefaultAssignmentStrategy(gamma)
    )
    db = TrainingDB(host=Environment.host_redis)
    agent = create_model(0)
    training_runtime = TrainingRunTime(
        Board(size),
        agent,
        value_assignment_strategy=value_assignment_strategy,
    )
    tournament_runtime = TournamentRuntime(size, db, max_game_length=max_game_length)

    # the trainer needs to start and flush the db, so we wait
    time.sleep(10)  # TODO: figure out why workers can start before trainer
    while True:
        while (pairing := db.tournament_get_match()) is not None:
            # if a tournament is on, we partake
            player_1, player_2 = pairing
            tournament_runtime.play_and_record_game(
                {
                    player_1: create_model(player_1),
                    player_2: create_model(player_2),
                }
            )
        traj = training_runtime.play_game(max_game_length=max_game_length)
        db.trajectory_post(traj)
