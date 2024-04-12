from dataclasses import dataclass

import numpy as np

from alpha_cc.agents.state import GameState


@dataclass
class MCTSExperience:
    state: GameState
    pi: np.ndarray
    v_target: np.floating | float = 0.0
