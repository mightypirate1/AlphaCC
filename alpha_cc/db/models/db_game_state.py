from dataclasses import dataclass
from typing import Self

from alpha_cc.agents.mcts import MCTSNodePy
from alpha_cc.engine import Board, Move
from alpha_cc.state import GameState
from alpha_cc.state.game_state import StateHash


@dataclass
class DBGameState:
    states: list[GameState]
    move_idxs: list[int]
    nodes: dict[StateHash, MCTSNodePy]

    @property
    def moves(self) -> list[Move]:
        return [state.board.get_moves()[idx] for state, idx in zip(self.states, self.move_idxs)]

    @property
    def state(self) -> GameState:
        return self.states[-1]

    @classmethod
    def new(cls: type[Self], size: int) -> Self:
        return cls(states=[GameState(Board(size))], move_idxs=[], nodes={})
