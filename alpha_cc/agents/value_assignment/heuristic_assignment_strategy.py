from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_experience import Experience, ProcessedExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import (
    _DEFAULT_WDL_WEIGHTS,
    DefaultAssignmentStrategy,
    WDLWeights,
    _normalize_wdl_weights,
    _weighted_wdl_sum,
)
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy
from alpha_cc.engine import Board


def _value_to_wdl(v: float) -> tuple[float, float, float]:
    """Convert a scalar value in [-1, 1] to a WDL tuple with zero draw weight."""
    v = max(-1.0, min(1.0, v))
    return ((1.0 + v) / 2.0, 0.0, (1.0 - v) / 2.0)


class HeuristicAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(
        self,
        size: int,
        heuristic_scale: float = 1.0,
        gamma: float = 1.0,
        wdl_weights: WDLWeights = _DEFAULT_WDL_WEIGHTS,
        wdl_smoothing: float = 0.0,
    ) -> None:
        self._heuristic = Heuristic(size, scale=heuristic_scale, subtract_opponent=True)
        self._default_assignment_strategy = DefaultAssignmentStrategy(
            gamma,
            wdl_weights=wdl_weights,
            wdl_smoothing=wdl_smoothing,
        )
        self._w_game, self._w_mcts, self._w_greedy = _normalize_wdl_weights(wdl_weights)
        self._wdl_smoothing = wdl_smoothing

    def __call__(self, trajectory: list[Experience], final_board: Board) -> list[ProcessedExperience]:
        if final_board.info.game_over:
            return self._default_assignment_strategy(trajectory, final_board)

        return [
            ProcessedExperience(
                state=exp.state,
                pi_target=exp.result.pi,
                wdl_target=_weighted_wdl_sum(
                    _value_to_wdl(self._heuristic(exp.state)),
                    exp.result,
                    self._w_game,
                    self._w_mcts,
                    self._w_greedy,
                    self._wdl_smoothing,
                ),
                game_ended_early=True,
            )
            for exp in trajectory
        ]
