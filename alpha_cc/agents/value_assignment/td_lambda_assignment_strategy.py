from alpha_cc.agents.mcts.mcts_experience import Experience, ProcessedExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import (
    WDLWeights,
    _DEFAULT_WDL_WEIGHTS,
    _flip_wdl,
    _mix_toward_draw,
    _normalize_wdl_weights,
    _weighted_wdl_sum,
)
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


def _wdl_lerp(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Linear interpolation: (1-t)*a + t*b."""
    return (
        (1 - t) * a[0] + t * b[0],
        (1 - t) * a[1] + t * b[1],
        (1 - t) * a[2] + t * b[2],
    )


class TDLambdaAssignmentStrategy(ValueAssignmentStrategy):
    """Value assignment using TD(lambda) blending of MCTS search WDL with game outcome.

    For each position, the target is a blend of:
      - The MCTS search-derived soft WDL (from Bayesian-blended node counts)
      - The game outcome, propagated backward with gamma discounting

    lambda_=0 gives pure MCTS targets, lambda_=1 gives pure game outcome.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_: float = 0.8,
        non_terminal_weight: float = 0.2,
        wdl_weights: WDLWeights = _DEFAULT_WDL_WEIGHTS,
        wdl_smoothing: float = 0.0,
    ) -> None:
        self._gamma = gamma
        self._lambda = lambda_
        self._non_terminal_weight = non_terminal_weight
        self._w_game, self._w_mcts, self._w_greedy = _normalize_wdl_weights(wdl_weights)
        self._wdl_smoothing = wdl_smoothing

    def __call__(self, trajectory: list[Experience], final_board: Board) -> list[ProcessedExperience]:
        game_wdl = _flip_wdl(final_board.info.wdl)
        weight = 1.0 if final_board.info.game_over else self._non_terminal_weight
        game_ended_early = not final_board.info.game_over

        next_wdl = game_wdl
        processed = []
        for exp in reversed(trajectory):
            mcts_wdl = (exp.result.mcts_wdl[0], exp.result.mcts_wdl[1], exp.result.mcts_wdl[2])
            base_wdl = _wdl_lerp(mcts_wdl, next_wdl, self._lambda)
            processed.append(
                ProcessedExperience(
                    state=exp.state,
                    pi_target=exp.result.pi,
                    wdl_target=_weighted_wdl_sum(
                        base_wdl, exp.result, self._w_game, self._w_mcts, self._w_greedy, self._wdl_smoothing,
                    ),
                    weight=weight,
                    game_ended_early=game_ended_early,
                )
            )
            next_wdl = _mix_toward_draw(_flip_wdl(base_wdl), self._gamma)
        processed.reverse()
        return processed
