import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import Board


class DummyAgent(Agent):
    def __init__(self) -> None:
        pass

    def choose_move(self, board: Board, training: bool = False) -> int:  # noqa: ARG002
        return int(np.random.choice(len(board.get_next_states())))

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass
