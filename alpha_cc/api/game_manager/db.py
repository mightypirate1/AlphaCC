from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from alpha_cc.agents.mcts import MCTSNodePy
from alpha_cc.engine import Move
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
    def from_state(cls: type[Self], state: GameState) -> Self:
        return cls(states=[state], move_idxs=[], nodes={})


class DB(ABC):
    @abstractmethod
    def get_entry(self, game_id: str) -> DBGameState:
        pass

    @abstractmethod
    def list_entries(self) -> list[str]:
        pass

    @abstractmethod
    def set_entry(self, game_id: str, state: DBGameState) -> None:
        pass

    @abstractmethod
    def remove_entry(self, game_id: str) -> bool:
        pass
