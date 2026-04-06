from math import ceil

import numpy as np
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.value_assignment import ValueAssignmentStrategy
from alpha_cc.engine import Board
from alpha_cc.state import GameState


class TrainingRunTime:
    def __init__(
        self,
        board: Board,
        value_assignment_strategy: ValueAssignmentStrategy,
    ) -> None:
        self._board = board
        self._value_assignment_strategy = value_assignment_strategy

    def play_game(
        self,
        agent: MCTSAgent,
        n_rollouts: int,
        rollout_depth: int,
        action_temperature: float = 1.0,
        max_game_length: int | None = None,
        argmax_delay: int | None = None,
        internal_nodes_fraction: float | None = None,
        internal_nodes_min_visits: int = 1,
    ) -> TrainingData:
        board = self._board.reset()
        max_game_duration = np.inf if max_game_length is None else max_game_length
        time_to_argmax = argmax_delay if argmax_delay is not None else np.inf
        agent.on_game_start()
        snapshot_interval = 1 if internal_nodes_fraction else 0

        trajectory: list[MCTSExperience] = []
        internal_nodes: dict[GameState, MCTSNodePy] = {}

        with tqdm(desc="training", total=max_game_length) as pbar:
            while not board.info.game_over:  # main terminaltion condition
                if board.info.duration >= max_game_duration:  # ended early termination condition
                    for exp in trajectory:
                        exp.game_ended_early = True
                    break

                pi, value = agent.run_rollouts(
                    board,
                    n_rollouts=n_rollouts,
                    rollout_depth=rollout_depth,
                    temperature=action_temperature,
                )
                # WDL placeholder — gets overwritten by value assignment strategy
                wdl_placeholder = ((1 + value) / 2, 0.0, (1 - value) / 2)
                experience = MCTSExperience(
                    state=GameState(board),
                    pi_target=pi,
                    wdl_target=wdl_placeholder,
                )
                trajectory.append(experience)

                if snapshot_interval and len(trajectory) % snapshot_interval == 0:
                    internal_nodes.update(
                        _sample_internal_nodes(
                            nodes=agent.internal_nodes,
                            trajectory=trajectory,
                            fraction=internal_nodes_fraction,  # type: ignore[arg-type]
                            min_visits=internal_nodes_min_visits,
                            already_have=len(internal_nodes),
                        )
                    )

                a = int(np.argmax(pi))
                if (time_to_argmax := time_to_argmax - 1) >= 0:
                    a = np.random.choice(len(pi), p=pi)

                board = board.apply(board.get_moves()[a])
                agent.on_move_applied(board)
                pbar.update(1)
        training_data = TrainingData(
            trajectory=self._value_assignment_strategy(trajectory, final_board=board),
            internal_nodes=internal_nodes,
            worker_stats=agent.get_worker_stats(),
            winner=board.info.winner,
        )
        agent.on_game_end()
        return training_data


def _sample_internal_nodes(
    nodes: dict[Board, MCTSNodePy],
    trajectory: list[MCTSExperience],
    fraction: float,
    min_visits: int,
    already_have: int,
) -> dict[GameState, MCTSNodePy]:
    """Sample internal nodes from the live MCTS cache.

    Returns up to ceil(len(trajectory) * fraction) - already_have new nodes
    that meet the min_visits threshold and are not already in the trajectory.
    """
    n_real = len(trajectory)
    needed = ceil(n_real * fraction) - already_have
    if needed <= 0:
        return {}

    real_hashes = {exp.state.hash for exp in trajectory}
    candidates = [
        (GameState(board), node)
        for board, node in nodes.items()
        if node.n.sum() >= min_visits and hash(board) not in real_hashes
    ]
    if not candidates:
        return {}

    if len(candidates) <= needed:
        return dict(candidates)

    candidates.sort(key=lambda sn: sn[1].n.sum(), reverse=True)
    return dict(candidates[:needed])
