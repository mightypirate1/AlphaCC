from dataclasses import dataclass

import numpy as np

from alpha_cc.engine import Move


@dataclass
class MCTSNode:
    pi: np.ndarray  # this is the nn-output; not mcts pi
    v_hat: float
    n: np.ndarray
    q: np.ndarray
    moves: list[Move]
