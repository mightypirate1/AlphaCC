"""
ValueAssignmentStrategy takes a trajectory of Experience and produces ProcessedExperience
with value targets assigned.
"""

from abc import ABC, abstractmethod

from alpha_cc.agents.mcts.mcts_experience import Experience, ProcessedExperience
from alpha_cc.engine import Board


class ValueAssignmentStrategy(ABC):
    @abstractmethod
    def __call__(self, trajectory: list[Experience], final_board: Board) -> list[ProcessedExperience]: ...


class NoOpAssignmentStrategy(ValueAssignmentStrategy):
    def __call__(self, trajectory: list[Experience], final_board: Board) -> list[ProcessedExperience]:
        game_ended_early = not final_board.info.game_over
        return [
            ProcessedExperience(
                state=exp.state,
                pi_target=exp.result.pi,
                wdl_target=(exp.result.mcts_wdl[0], exp.result.mcts_wdl[1], exp.result.mcts_wdl[2]),
                game_ended_early=game_ended_early,
            )
            for exp in trajectory
        ]
