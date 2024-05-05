from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


class DefaultAssignmentStrategyWithHeuristic(ValueAssignmentStrategy):
    def __init__(self, size: int, heuristic_scale: float = 1.0, gamma: float = 1.0) -> None:
        self._heuristic = Heuristic(size, scale=heuristic_scale, subtract_opponent=True)
        self._gamma = gamma

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        last_exp = trajectory[-1]
        value = self._heuristic(last_exp.state)
        if final_board.info.game_over:
            # the final board is the board AFTER the board on the last experience in the trajectory
            # this, it's viewed from the perspective of the other player, and we need to multiply
            # by -1 to get the correct reward
            value = -float(final_board.info.reward)

        last_exp.v_target = value

        for exp in reversed(trajectory):
            exp.v_target = value
            value *= -self._gamma
        return trajectory
