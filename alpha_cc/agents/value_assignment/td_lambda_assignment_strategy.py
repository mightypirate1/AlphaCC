from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.default_assignment_strategy import _flip_wdl, _mix_toward_draw
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

    def __init__(self, gamma: float = 1.0, lambda_: float = 0.8, non_terminal_weight: float = 0.2) -> None:
        self._gamma = gamma
        self._lambda = lambda_
        self._non_terminal_weight = non_terminal_weight

    def __call__(self, trajectory: list[MCTSExperience], final_board: Board) -> list[MCTSExperience]:
        game_wdl = _flip_wdl(final_board.info.wdl)
        weight = 1.0 if final_board.info.game_over else self._non_terminal_weight

        # TD(lambda) backward pass
        next_wdl = game_wdl
        for exp in reversed(trajectory):
            exp.weight = weight
            if exp.mcts_wdl is not None:
                exp.wdl_target = _wdl_lerp(exp.mcts_wdl, next_wdl, self._lambda)
            else:
                exp.wdl_target = next_wdl
            next_wdl = _mix_toward_draw(_flip_wdl(exp.wdl_target), self._gamma)

        return trajectory
