from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import DefaultAssignmentStrategy
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


class HeuristicAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(self, size: int, heuristic_scale: float = 1.0, gamma: float = 1.0) -> None:
        self._heuristic = Heuristic(size, scale=heuristic_scale, subtract_opponent=True)
        self._default_assignment_strategy = DefaultAssignmentStrategy(gamma)

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        if final_board.info.game_over:
            return self._default_assignment_strategy(trajectory, final_board)

        for exp in reversed(trajectory):
            exp.v_target = self._heuristic(exp.state)
        return trajectory
