import numpy as np

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.state import GameState
from alpha_cc.nn.nets.dual_head_net import DualHeadNet
from alpha_cc.reward import HeuristicReward


class BogusNet(DualHeadNet[list[list[MCTSExperience]]]):
    heuristic = HeuristicReward(9)

    def policy(self, state: GameState) -> np.ndarray:
        pi = np.array([self.heuristic(sp) for sp in state.children])
        pi = pi / pi.sum()
        return pi

    def value(self, state: GameState) -> np.floating:
        return self.heuristic(state)

    def update_weights(self, train_data: list[list[MCTSExperience]]) -> None:
        print(f"pretend training on {len(train_data)} trajectories")  # noqa
