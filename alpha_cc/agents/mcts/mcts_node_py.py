from dataclasses import dataclass

import numpy as np


@dataclass
class MCTSNodePy:
    pi: np.ndarray  # this is the nn-output; not mcts pi
    v_hat: float | np.floating
    n: np.ndarray
    q: np.ndarray
