from collections.abc import Iterable

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store.node_store import NodeStore
from alpha_cc.engine import Board


class LocalNodeStore(NodeStore):
    def __init__(self) -> None:
        self._store: dict[Board, MCTSNodePy] = {}

    def __contains__(self, board: Board) -> bool:
        return board in self._store

    def keys(self) -> Iterable[Board]:
        return self._store.keys()

    def get(self, board: Board) -> MCTSNodePy:
        return self._store[board]

    def set(self, board: Board, node: MCTSNodePy) -> None:
        self._store[board] = node

    def clear(self) -> None:
        return self._store.clear()

    def load_from(self, node_store: NodeStore) -> None:
        return self._store.update({k: node_store.get(k) for k in node_store.keys()})  # noqa
