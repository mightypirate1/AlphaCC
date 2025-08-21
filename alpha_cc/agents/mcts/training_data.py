from dataclasses import dataclass

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.state import GameState


@dataclass
class TrainingData:
    trajectory: list[MCTSExperience]
    internal_nodes: dict[GameState, MCTSNodePy]

    def __bool__(self) -> bool:
        return bool(self.trajectory)
