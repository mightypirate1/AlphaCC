from __future__ import annotations

from dataclasses import dataclass, field

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.state import GameState


@dataclass
class TrainingData:
    trajectory: list[MCTSExperience]
    internal_nodes: dict[GameState, MCTSNodePy]
    worker_stats: WorkerStats = field(default_factory=WorkerStats.empty)

    def __bool__(self) -> bool:
        return bool(self.trajectory)
