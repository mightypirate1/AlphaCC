from alpha_cc.agents.mcts.mcts_experience import Experience, ProcessedExperience
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board, RolloutResult

# WDL representing pure uncertainty (draw).
_DRAW_WDL = (0.0, 1.0, 0.0)

WDLWeights = tuple[float, float, float]  # (game, mcts, greedy)
_DEFAULT_WDL_WEIGHTS: WDLWeights = (1.0, 0.0, 0.0)


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


def _normalize_wdl_weights(weights: WDLWeights) -> WDLWeights:
    total = sum(weights)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _smooth_wdl(wdl: tuple[float, float, float], epsilon: float) -> tuple[float, float, float]:
    """Label smoothing: (1 - epsilon) * wdl + epsilon * uniform(1/3)."""
    u = epsilon / 3.0
    r = 1.0 - epsilon
    return (r * wdl[0] + u, r * wdl[1] + u, r * wdl[2] + u)


def _weighted_wdl_sum(
    base: tuple[float, float, float],
    result: RolloutResult,
    w_game: float,
    w_mcts: float,
    w_greedy: float,
    smoothing: float,
) -> tuple[float, float, float]:
    """Blend base WDL with search estimates, then apply label smoothing."""
    mcts = result.mcts_wdl
    greedy = result.greedy_backup_wdl
    wdl = (
        w_game * base[0] + w_mcts * mcts[0] + w_greedy * greedy[0],
        w_game * base[1] + w_mcts * mcts[1] + w_greedy * greedy[1],
        w_game * base[2] + w_mcts * mcts[2] + w_greedy * greedy[2],
    )
    if smoothing > 0:
        wdl = _smooth_wdl(wdl, smoothing)
    return wdl


class DefaultAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(
        self,
        gamma: float = 1.0,
        non_terminal_weight: float = 0.2,
        wdl_weights: WDLWeights = _DEFAULT_WDL_WEIGHTS,
        wdl_smoothing: float = 0.0,
    ) -> None:
        self._gamma = gamma
        self._non_terminal_weight = non_terminal_weight
        self._w_game, self._w_mcts, self._w_greedy = _normalize_wdl_weights(wdl_weights)
        self._wdl_smoothing = wdl_smoothing

    def __call__(self, trajectory: list[Experience], final_board: Board) -> list[ProcessedExperience]:
        wdl = self._wdl_when_not_game_over(final_board)
        weight = self._non_terminal_weight
        if final_board.info.game_over:
            wdl = _flip_wdl(final_board.info.wdl)
            weight = 1.0

        game_ended_early = not final_board.info.game_over
        processed = []
        for exp in reversed(trajectory):
            processed.append(
                ProcessedExperience(
                    state=exp.state,
                    pi_target=exp.result.pi,
                    wdl_target=_weighted_wdl_sum(
                        wdl,
                        exp.result,
                        self._w_game,
                        self._w_mcts,
                        self._w_greedy,
                        self._wdl_smoothing,
                    ),
                    weight=weight,
                    game_ended_early=game_ended_early,
                )
            )
            wdl = _mix_toward_draw(_flip_wdl(wdl), self._gamma)
        processed.reverse()
        return processed

    def _wdl_when_not_game_over(self, final_board: Board) -> tuple[float, float, float]:
        return _flip_wdl(final_board.info.wdl)
