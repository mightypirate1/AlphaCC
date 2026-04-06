from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board

# WDL representing pure uncertainty (draw).
_DRAW_WDL = (0.0, 1.0, 0.0)


def _flip_wdl(wdl: tuple[float, float, float]) -> tuple[float, float, float]:
    """Swap win and loss to change player perspective."""
    return (wdl[2], wdl[1], wdl[0])


def _mix_toward_draw(wdl: tuple[float, float, float], gamma: float) -> tuple[float, float, float]:
    """Discount WDL by mixing toward draw (uncertainty). gamma=1 keeps unchanged."""
    return (
        gamma * wdl[0] + (1 - gamma) * _DRAW_WDL[0],
        gamma * wdl[1] + (1 - gamma) * _DRAW_WDL[1],
        gamma * wdl[2] + (1 - gamma) * _DRAW_WDL[2],
    )


class DefaultAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(self, gamma: float = 1.0, non_terminal_weight: float = 0.2) -> None:
        self._gamma = gamma
        self._non_terminal_weight = non_terminal_weight

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        wdl = self._wdl_when_not_game_over(final_board)
        weight = self._non_terminal_weight
        if final_board.info.game_over:
            # the final board is viewed from the perspective of the other player,
            # so we flip W/L to get the correct perspective
            wdl = _flip_wdl(final_board.info.wdl)
            weight = 1.0

        for exp in reversed(trajectory):
            exp.weight = weight
            exp.wdl_target = wdl
            wdl = _mix_toward_draw(_flip_wdl(wdl), self._gamma)
        return trajectory

    def _wdl_when_not_game_over(self, final_board: Board) -> tuple[float, float, float]:  # noqa
        return _flip_wdl(final_board.info.wdl)
