from abc import ABC, abstractmethod
from dataclasses import dataclass

from alpha_cc.agents.mcts import MCTSNodePy
from alpha_cc.state import GameState


@dataclass
class DBGameState:
    state: GameState
    nodes: list[MCTSNodePy]


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
