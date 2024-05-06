import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.engine import Board
from alpha_cc.state import GameState


class GreedyAgent(Agent):
    def __init__(self, board_size: int = 9) -> None:
        self._heuristic_function = Heuristic(board_size, subtract_opponent=True)

    def choose_move(self, board: Board, _: bool = False) -> int:
        sp_values = self._evaluation(board)
        action = int(np.argmax(sp_values))
        return action

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    def _evaluation(self, board: Board) -> np.ndarray:
        def heur(sp: GameState) -> float:
            # important we specify player, since we want the value for
            # the _next_ state but for the _current_ player
            return self._heuristic_function(sp, for_player=player)

        s = GameState(board)
        player = board.info.current_player
        return np.asarray([heur(sp) for sp in s.children])
