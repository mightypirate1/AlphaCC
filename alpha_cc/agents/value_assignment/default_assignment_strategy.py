from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


class DefaultAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(self, gamma: float = 1.0) -> None:
        self._gamma = gamma

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        value = self._value_when_not_game_over(trajectory, final_board)
        last_exp = trajectory[-1]
        if final_board.info.game_over:
            # the final board is the board AFTER the board on the last experience in the trajectory
            # thus, it's viewed from the perspective of the other player, and we need to multiply
            # by -1 to get the correct reward
            value = -final_board.info.reward
        last_exp.v_target = value

        for exp in reversed(trajectory):
            exp.v_target = value
            value *= -self._gamma
        return trajectory

    def _value_when_not_game_over(self, trajectory: list[MCTSExperience], final_board: Board) -> float:  # noqa
        return -final_board.info.reward
