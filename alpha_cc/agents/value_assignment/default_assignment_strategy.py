from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.value_assignment.value_assignment_strategy import ValueAssignmentStrategy


class DefaultAssignmentStrategy(ValueAssignmentStrategy):
    def __init__(self, gamma: float = 1.0) -> None:
        self._gamma = gamma

    def __call__(self, trajectory: list[MCTSExperience], final_state_value: float) -> list[MCTSExperience]:
        value = final_state_value
        for exp in reversed(trajectory):
            exp.v_target = value
            value *= -self._gamma
        return trajectory
