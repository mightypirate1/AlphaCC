from dataclasses import dataclass

import numpy as np


@dataclass
class MCTSNode:
    v_hat: float
    pi: np.ndarray  # this is the nn-output; not mcts pi
    n: np.ndarray
    q: np.ndarray
