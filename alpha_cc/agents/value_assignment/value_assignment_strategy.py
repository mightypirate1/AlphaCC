"""
ValueAssignmentStrategy takes a trajectory and modifies it to set value targets.
"""

from abc import ABC, abstractmethod

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience


class ValueAssignmentStrategy(ABC):
    @abstractmethod
    def __call__(self, trajectory: list[MCTSExperience], final_state_value: float) -> list[MCTSExperience]: ...


class NoOpAssignmentStrategy(ValueAssignmentStrategy):
    def __call__(self, trajectory: list[MCTSExperience], _: float) -> list[MCTSExperience]:
        return trajectory
