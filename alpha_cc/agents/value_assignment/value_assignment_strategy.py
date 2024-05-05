"""
ValueAssignmentStrategy takes a trajectory and modifies it to set value targets.
"""

from abc import ABC, abstractmethod

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.engine import Board


class ValueAssignmentStrategy(ABC):
    @abstractmethod
    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]: ...


class NoOpAssignmentStrategy(ValueAssignmentStrategy):
    def __call__(self, trajectory: list[MCTSExperience], _: Board) -> list[MCTSExperience]:
        return trajectory
