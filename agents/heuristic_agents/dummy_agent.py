from typing import List
import numpy as np
from alpha_cc import Board, BoardInfo
from agents.base_agent import BaseAgent, GameState

class DummyAgent(BaseAgent):
    def __init__(self):
        pass

    def choose_action(self, board: Board) -> int:
        s_primes = self.unpack_s_primes(board)
        action = np.random.choice(len(s_primes))
        return action
