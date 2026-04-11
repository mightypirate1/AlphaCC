import logging
import signal
import sys
import time
from typing import Any

import click

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    HeuristicAssignmentStrategy,
    TDLambdaAssignmentStrategy,
    ValueAssignmentStrategy,
)
from alpha_cc.config import Environment
from alpha_cc.db import GamesDB, TrainingDB
from alpha_cc.engine import Board
from alpha_cc.logs import init_rootlogger
from alpha_cc.runtimes.tournament_runtime import TournamentRuntime
from alpha_cc.runtimes.training_runtime import TrainingRunTime
from alpha_cc.utils.param_schedule import ParamSchedule

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=str, default="100")
@click.option("--rollout-depth", type=str, default="100")
@click.option("--max-game-length", type=str)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--dirichlet-noise-weight", type=float, default=0.0)
@click.option("--dirichlet-leaf-noise-weight", type=str, default="0.0")
@click.option("--argmax-delay", type=str, default=None)
@click.option("--action-temperature", type=str, default="1.0")
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--non-terminal-value-weight", type=float, default=0.1)
@click.option("--wdl-gamma", type=float, default=1.0, help="Gamma discount for backward WDL propagation.")
@click.option(
    "--wdl-lambda", type=float, default=None, help="TD(lambda) blending factor. Enables soft MCTS WDL targets."
)
@click.option("--wdl-weight-game", type=float, default=1.0, help="Weight for game-outcome WDL in target blend.")
@click.option("--wdl-weight-mcts", type=float, default=0.0, help="Weight for MCTS root WDL in target blend.")
@click.option("--wdl-weight-greedy", type=float, default=0.0, help="Weight for greedy-backup WDL in target blend.")
@click.option("--wdl-smoothing", type=float, default=0.0, help="Label smoothing epsilon for WDL targets.")
@click.option("--internal-nodes-fraction", type=str, default="0.0")
@click.option("--internal-nodes-min-visits", type=str, default="1")
@click.option("--n-threads", type=int, default=1)
@click.option("--pruning-tree", is_flag=True, default=False)
@click.option("--debug-prints", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: str,
    rollout_depth: str,
    max_game_length: str | None,
    rollout_gamma: float,
    dirichlet_noise_weight: float,
    dirichlet_leaf_noise_weight: str,
    argmax_delay: str | None,
    action_temperature: str,
    heuristic: bool,
    non_terminal_value_weight: float,
    wdl_gamma: float,
    wdl_lambda: float | None,
    wdl_weight_game: float,
    wdl_weight_mcts: float,
    wdl_weight_greedy: float,
    wdl_smoothing: float,
    internal_nodes_fraction: str,
    internal_nodes_min_visits: str,
    n_threads: int,
    pruning_tree: bool,
    debug_prints: bool,
    verbose: bool,
) -> None:
    def create_model(channel: int, trainer_time: int, dummy_preds: bool) -> MCTSAgent:
        return MCTSAgent(
            nn_service_addr=Environment.nn_service_addr,
            pred_channel=channel,
            n_rollouts=n_rollouts_schedule.as_int(trainer_time),
            rollout_depth=rollout_depth_schedule.as_int(trainer_time),
            rollout_gamma=rollout_gamma,
            dirichlet_weight=dirichlet_noise_weight,
            dirichlet_leaf_weight=dirichlet_leaf_noise_weight_schedule.as_float(trainer_time),
            n_threads=n_threads,
            pruning_tree=pruning_tree,
            debug_prints=debug_prints,
            dummy_preds=dummy_preds,
        )

    max_game_length_schedule = ParamSchedule.from_str(max_game_length or "inf")
    argmax_delay_schedule = ParamSchedule.from_str(argmax_delay or "inf")
    n_rollouts_schedule = ParamSchedule.from_str(n_rollouts)
    rollout_depth_schedule = ParamSchedule.from_str(rollout_depth)
    action_temperature_schedule = ParamSchedule.from_str(action_temperature)
    internal_nodes_fraction_schedule = ParamSchedule.from_str(internal_nodes_fraction)
    internal_nodes_min_visits_schedule = ParamSchedule.from_str(internal_nodes_min_visits)
    dirichlet_leaf_noise_weight_schedule = ParamSchedule.from_str(dirichlet_leaf_noise_weight)

    init_rootlogger(verbose=verbose)
    create_and_register_signal_handler()
    training_db = TrainingDB(host=Environment.redis_host_main)
    games_db = GamesDB(host=Environment.redis_host_main)

    wdl_weights = (wdl_weight_game, wdl_weight_mcts, wdl_weight_greedy)
    value_assignment_strategy = create_value_assignment_strategy(
        size, wdl_gamma, heuristic, non_terminal_value_weight, wdl_lambda, wdl_weights, wdl_smoothing
    )
    training_runtime = TrainingRunTime(Board(size), value_assignment_strategy)
    tournament_runtime = TournamentRuntime(
        size, training_db, games_db, max_game_length=max_game_length_schedule.as_int(0)
    )

    # the trainer needs to start and flush the db, so we wait
    time.sleep(10)  # TODO: figure out why workers can start before trainer
    while True:
        trainer_time = training_db.weights_fetch_latest_index()
        while (pairing := training_db.tournament_get_match()) is not None:
            # if a tournament is on, we partake
            player_1, player_2 = pairing
            tournament_runtime.play_and_record_game(
                {
                    player_1: create_model(player_1, trainer_time, False),
                    player_2: create_model(player_2, trainer_time, False),
                }
            )
        internal_nodes_fraction_val = internal_nodes_fraction_schedule.as_float(trainer_time)
        training_data = training_runtime.play_game(
            agent=create_model(0, trainer_time, training_db.nn_warmup_get() < 0),
            n_rollouts=n_rollouts_schedule.as_int(trainer_time),
            rollout_depth=rollout_depth_schedule.as_int(trainer_time),
            max_game_length=max_game_length_schedule.as_int(trainer_time),
            action_temperature=action_temperature_schedule.as_float(trainer_time),
            argmax_delay=argmax_delay_schedule.as_int(trainer_time),
            internal_nodes_fraction=internal_nodes_fraction_val if internal_nodes_fraction_val > 0.0 else None,
            internal_nodes_min_visits=internal_nodes_min_visits_schedule.as_int(trainer_time),
        )
        training_db.training_data_post(training_data)


def create_value_assignment_strategy(
    size: int,
    wdl_gamma: float,
    heuristic: bool,
    non_terminal_weight: float,
    wdl_lambda: float | None = None,
    wdl_weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    wdl_smoothing: float = 0.0,
) -> ValueAssignmentStrategy:
    if wdl_lambda is not None:
        return TDLambdaAssignmentStrategy(
            gamma=wdl_gamma,
            lambda_=wdl_lambda,
            non_terminal_weight=non_terminal_weight,
            wdl_weights=wdl_weights,
            wdl_smoothing=wdl_smoothing,
        )
    if heuristic:
        return HeuristicAssignmentStrategy(
            size,
            gamma=wdl_gamma,
            wdl_weights=wdl_weights,
            wdl_smoothing=wdl_smoothing,
        )
    return DefaultAssignmentStrategy(
        wdl_gamma,
        non_terminal_weight=non_terminal_weight,
        wdl_weights=wdl_weights,
        wdl_smoothing=wdl_smoothing,
    )


def create_and_register_signal_handler() -> None:
    def signal_handler(signum: int, _: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
