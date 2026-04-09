from dataclasses import dataclass

import numpy as np

from alpha_cc.engine import RolloutResult
from alpha_cc.state import GameState


@dataclass
class Experience:
    state: GameState
    result: RolloutResult


@dataclass
class ProcessedExperience:
    state: GameState
    pi_target: np.ndarray
    wdl_target: tuple[float, float, float]
    weight: float = 1.0
    is_internal_node: bool = False
    game_ended_early: bool = False
