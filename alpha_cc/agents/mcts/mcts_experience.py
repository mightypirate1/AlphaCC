from dataclasses import dataclass

import numpy as np

from alpha_cc.state import GameState


@dataclass
class MCTSExperience:
    state: GameState
    pi_target: np.ndarray
    v_target: float  # may be (re-)assigned after end of game
    weight: float = 1.0
