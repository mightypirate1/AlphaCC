from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.state.game_state import StateHash


class NodeStore(ABC):
    @abstractmethod
    def __contains__(self, state_hash: StateHash) -> bool:
        pass

    @abstractmethod
    def keys(self) -> Iterable[StateHash]:
        pass

    @abstractmethod
    def get(self, state_hash: StateHash) -> MCTSNodePy:
        pass

    @abstractmethod
    def set(self, state_hash: StateHash, node: MCTSNodePy) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def load_from(self, node_store: NodeStore) -> None:
        pass
