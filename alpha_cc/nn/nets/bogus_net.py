import numpy as np

from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.nn.nets.dual_head_net import DualHeadNet
from alpha_cc.state import GameState


class BogusNet(DualHeadNet):
    def __init__(self, board_size: int) -> None:
        self._heuristic = Heuristic(board_size)

    def policy(self, state: GameState) -> np.ndarray:
        pi = np.array([self._heuristic(sp) for sp in state.children])
        pi = pi / pi.sum()
        return pi

    def value(self, state: GameState) -> np.floating:
        return np.float32(self._heuristic(state))
