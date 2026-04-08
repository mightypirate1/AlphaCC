from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import DefaultAssignmentStrategy
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


def _value_to_wdl(v: float) -> tuple[float, float, float]:
    """Convert a scalar value in [-1, 1] to a WDL tuple with zero draw weight."""
    v = max(-1.0, min(1.0, v))
    return ((1.0 + v) / 2.0, 0.0, (1.0 - v) / 2.0)


class HeuristicAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(self, size: int, heuristic_scale: float = 1.0, gamma: float = 1.0) -> None:
        self._heuristic = Heuristic(size, scale=heuristic_scale, subtract_opponent=True)
        self._default_assignment_strategy = DefaultAssignmentStrategy(gamma)

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        if final_board.info.game_over:
            return self._default_assignment_strategy(trajectory, final_board)

        for exp in reversed(trajectory):
            exp.wdl_target = _value_to_wdl(self._heuristic(exp.state))
        return trajectory
