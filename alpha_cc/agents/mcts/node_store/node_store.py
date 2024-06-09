from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.engine import Board


class NodeStore(ABC):
    @abstractmethod
    def __contains__(self, board: Board) -> bool:
        pass

    @abstractmethod
    def keys(self) -> Iterable[Board]:
        pass

    @abstractmethod
    def get(self, board: Board) -> MCTSNodePy:
        pass

    @abstractmethod
    def set(self, board: Board, node: MCTSNodePy) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def fetch_updated(self, clear: bool = True) -> dict[Board, MCTSNodePy]:
        pass

    @abstractmethod
    def load_from(self, node_store: NodeStore | dict[Board, MCTSNodePy]) -> None:
        pass
