import logging
import signal
import sys
import time
from math import ceil
from typing import Any

import click

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
    HeuristicAssignmentStrategy,
    ValueAssignmentStrategy,
)
from alpha_cc.config import Environment
from alpha_cc.db import GamesDB, TrainingDB
from alpha_cc.engine import Board
from alpha_cc.logs import init_rootlogger
from alpha_cc.runtimes.tournament_runtime import TournamentRuntime
from alpha_cc.runtimes.training_runtime import TrainingRunTime
from alpha_cc.state.game_state import GameState
from alpha_cc.utils.param_schedule import ParamSchedule

logger = logging.getLogger(__file__)


@click.command("alpha-cc-worker")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=str, default="100")
@click.option("--rollout-depth", type=str, default="100")
@click.option("--max-game-length", type=str)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--dirichlet-noise-weight", type=float, default=0.0)
@click.option("--argmax-delay", type=str, default=None)
@click.option("--action-temperature", type=str, default="1.0")
@click.option("--gamma", type=float, default=1.0)
@click.option("--heuristic", is_flag=True, default=False)
@click.option("--non-terminal-value-weight", type=float, default=0.1)
@click.option("--internal-nodes-fraction", type=str, default="0.0")
@click.option("--internal-nodes-min-visits", type=str, default="1")
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    n_rollouts: str,
    rollout_depth: str,
    max_game_length: str | None,
    rollout_gamma: float,
    dirichlet_noise_weight: float,
    argmax_delay: str | None,
    action_temperature: str,
    gamma: float,
    heuristic: bool,
    non_terminal_value_weight: float,
    internal_nodes_fraction: str,
    internal_nodes_min_visits: str,
    verbose: bool,
) -> None:
    def create_model(channel: int, trainer_time: int) -> MCTSAgent:
        return MCTSAgent(
            zmq_url=Environment.zmq_url,
            memcached_host=Environment.memcached_host,
            pred_channel=channel,
            n_rollouts=n_rollouts_schedule.as_int(trainer_time),
            rollout_depth=rollout_depth_schedule.as_int(trainer_time),
            rollout_gamma=rollout_gamma,
            dirichlet_weight=dirichlet_noise_weight,
        )

    max_game_length_schedule = ParamSchedule.from_str(max_game_length or "inf")
    argmax_delay_schedule = ParamSchedule.from_str(argmax_delay or "inf")
    n_rollouts_schedule = ParamSchedule.from_str(n_rollouts)
    rollout_depth_schedule = ParamSchedule.from_str(rollout_depth)
    action_temperature_schedule = ParamSchedule.from_str(action_temperature)
    internal_nodes_fraction_schedule = ParamSchedule.from_str(internal_nodes_fraction)
    internal_nodes_min_visits_schedule = ParamSchedule.from_str(internal_nodes_min_visits)

    init_rootlogger(verbose=verbose)
    create_and_register_signal_handler()
    training_db = TrainingDB(host=Environment.redis_host_main)
    games_db = GamesDB(host=Environment.redis_host_main)

    value_assignment_strategy = create_value_assignment_strategy(size, gamma, heuristic, non_terminal_value_weight)
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
                    player_1: create_model(player_1, trainer_time),
                    player_2: create_model(player_2, trainer_time),
                }
            )
        training_data = training_runtime.play_game(
            agent=create_model(0, trainer_time),
            max_game_length=max_game_length_schedule.as_int(trainer_time),
            action_temperature=action_temperature_schedule.as_float(trainer_time),
            argmax_delay=argmax_delay_schedule.as_int(trainer_time),
        )
        training_data.internal_nodes = filter_internal_nodes(
            training_data,
            internal_nodes_fraction=internal_nodes_fraction_schedule.as_float(trainer_time),
            internal_nodes_min_visits=internal_nodes_min_visits_schedule.as_int(trainer_time),
        )
        training_db.training_data_post(training_data)


def filter_internal_nodes(
    training_data: TrainingData, internal_nodes_fraction: float, internal_nodes_min_visits: int
) -> dict[GameState, MCTSNodePy]:
    """
    Select up to ceil(fraction * n_real) internal nodes whose visit counts
    are >= internal_nodes_min_visits. If fewer qualify, return them all.
    Guarantees we never exceed the target (ties are truncated).
    """
    if internal_nodes_fraction <= 0.0:
        return {}

    n_real = len(training_data.trajectory)
    real_hashes = {exp.state.hash for exp in training_data.trajectory}
    if n_real == 0:
        return {}

    target = ceil(n_real * internal_nodes_fraction)
    if target <= 0:
        return {}

    # Filter by min visits first
    candidates = [
        (state, node)
        for state, node in training_data.internal_nodes.items()
        if node.n.sum() >= internal_nodes_min_visits and state.hash not in real_hashes
    ]
    if not candidates:
        return {}

    if len(candidates) <= target:
        return dict(candidates)

    # Sort descending by visit count and take the top `target`
    candidates.sort(key=lambda sn: sn[1].n.sum(), reverse=True)
    top = candidates[:target]
    return dict(top)


def create_value_assignment_strategy(
    size: int, gamma: float, heuristic: bool, non_terminal_weight: float
) -> ValueAssignmentStrategy:
    if heuristic:
        return HeuristicAssignmentStrategy(size, gamma)
    return DefaultAssignmentStrategy(gamma, non_terminal_weight=non_terminal_weight)


def create_and_register_signal_handler() -> None:
    def signal_handler(signum: int, _: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
