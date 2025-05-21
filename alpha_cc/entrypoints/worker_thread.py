import logging
import time

import click

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    HeuristicAssignmentStrategy,
    ValueAssignmentStrategy,
)
from alpha_cc.config import Environment
from alpha_cc.db import GamesDB, TrainingDB
from alpha_cc.engine import Board
from alpha_cc.logs import init_rootlogger
from alpha_cc.runtimes import TournamentRuntime, TrainingRunTime
from alpha_cc.training import ParamSchedule

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=str, default="100")
@click.option("--rollout-depth", type=str, default="100")
@click.option("--max-game-length", type=int)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--dirichlet-noise-weight", type=float, default=0.0)
@click.option("--argmax-delay", type=str, default=None)
@click.option("--action-temperature", type=str, default="1.0")
@click.option("--gamma", type=float, default=1.0)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--non-terminal-value-weight", type=float, default=0.2)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: str,
    rollout_depth: str,
    max_game_length: int | None,
    rollout_gamma: float,
    dirichlet_noise_weight: float,
    argmax_delay: str | None,
    action_temperature: str,
    gamma: float,
    heuristic: bool,
    non_terminal_value_weight: float,
    verbose: bool,
) -> None:
    def create_model(channel: int, trainer_time: int) -> MCTSAgent:
        return MCTSAgent(
            redis_host=Environment.redis_host_pred,
            pred_channel=channel,
            n_rollouts=n_rollouts_schedule.as_int(trainer_time),
            rollout_depth=rollout_depth_schedule.as_int(trainer_time),
            rollout_gamma=rollout_gamma,
            dirichlet_weight=dirichlet_noise_weight,
        )

    argmax_delay_schedule = ParamSchedule.from_str(argmax_delay or "inf")
    n_rollouts_schedule = ParamSchedule.from_str(n_rollouts)
    rollout_depth_schedule = ParamSchedule.from_str(rollout_depth)
    action_temperature_schedule = ParamSchedule.from_str(action_temperature)

    init_rootlogger(verbose=verbose)
    training_db = TrainingDB(host=Environment.redis_host_main)
    games_db = GamesDB(host=Environment.redis_host_main)

    value_assignment_strategy = create_value_assignment_strategy(size, gamma, heuristic, non_terminal_value_weight)
    training_runtime = TrainingRunTime(Board(size), value_assignment_strategy)
    tournament_runtime = TournamentRuntime(size, training_db, games_db, max_game_length=max_game_length)

    # the trainer needs to start and flush the db, so we wait
    time.sleep(10)  # TODO: figure out why workers can start before trainer
    while True:
        trainer_time = training_db.weights_fetch_latest_index()
        while (pairing := training_db.tournament_get_match()) is not None:
            # if a tournament is on, we partake
            player_1, player_2 = pairing
            tournament_runtime.play_and_record_game(
                {
                    player_1: create_model(player_1, trainer_time),
                    player_2: create_model(player_2, trainer_time),
                }
            )
        traj = training_runtime.play_game(
            agent=create_model(0, trainer_time),
            max_game_length=max_game_length,
            action_temperature=action_temperature_schedule.as_float(trainer_time),
            argmax_delay=argmax_delay_schedule.as_int(0),
        )
        training_db.trajectory_post(traj)


def create_value_assignment_strategy(
    size: int, gamma: float, heuristic: bool, non_terminal_weight: float
) -> ValueAssignmentStrategy:
    if heuristic:
        return HeuristicAssignmentStrategy(size, gamma)
    return DefaultAssignmentStrategy(gamma, non_terminal_weight=non_terminal_weight)
