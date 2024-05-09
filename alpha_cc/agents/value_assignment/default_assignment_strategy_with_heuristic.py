from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import DefaultAssignmentStrategy


class DefaultAssignmentStrategyWithHeuristic(DefaultAssignmentStrategy):
    def __init__(self, size: int, heuristic_scale: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__(gamma)
        self._heuristic = Heuristic(size, scale=heuristic_scale, subtract_opponent=True)

    def _value_when_not_game_over(self, trajectory: list[MCTSExperience]) -> float:
        return self._heuristic(trajectory[-1].state)
