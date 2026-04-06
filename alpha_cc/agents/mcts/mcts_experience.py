from dataclasses import dataclass

import numpy as np

from alpha_cc.state import GameState


@dataclass
class MCTSExperience:
    state: GameState
    pi_target: np.ndarray
    wdl_target: tuple[float, float, float]  # (win, draw, loss) from current player's perspective; may be (re-)assigned after end of game
    weight: float = 1.0
    is_internal_node: bool = False
    game_ended_early: bool = False  # Needs to be set by the runtime
