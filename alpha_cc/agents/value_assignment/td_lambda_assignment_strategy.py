from alpha_cc.agents.mcts.mcts_experience import Experience, ProcessedExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import (
    _DEFAULT_WDL_WEIGHTS,
    WDLWeights,
    _flip_wdl,
    _mix_toward_draw,
    _normalize_wdl_weights,
    _smooth_wdl,
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
    """Value assignment using TD(lambda) blending of search estimates with game outcome.

    At each position (iterating backward from the end):
      1. Compute a weighted sum of the propagated return, mcts WDL, and greedy WDL
      2. Blend that with the propagated return via lambda
      3. Apply label smoothing
      4. Propagate the blended result backward (flip + gamma discount)

    lambda_=0 gives pure weighted estimate, lambda_=1 gives pure game outcome.
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
        v = _flip_wdl(final_board.info.wdl)
        weight = 1.0 if final_board.info.game_over else self._non_terminal_weight
        game_ended_early = not final_board.info.game_over

        processed = []
        for exp in reversed(trajectory):
            v_i = _weighted_wdl_sum(v, exp.result, self._w_game, self._w_mcts, self._w_greedy, 0.0)
            v_i = _wdl_lerp(v_i, v, self._lambda)
            if self._wdl_smoothing > 0:
                v_i = _smooth_wdl(v_i, self._wdl_smoothing)
            processed.append(
                ProcessedExperience(
                    state=exp.state,
                    pi_target=exp.result.pi,
                    wdl_target=v_i,
                    weight=weight,
                    game_ended_early=game_ended_early,
                )
            )
            v = _mix_toward_draw(_flip_wdl(v_i), self._gamma)
        processed.reverse()
        return processed
