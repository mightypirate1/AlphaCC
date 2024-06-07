from collections.abc import Iterable

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store.node_store import NodeStore
from alpha_cc.state.game_state import StateHash


class LocalNodeStore(NodeStore):
    def __init__(self) -> None:
        self._store: dict[StateHash, MCTSNodePy] = {}

    def __contains__(self, state_hash: StateHash) -> bool:
        return state_hash in self._store

    def keys(self) -> Iterable[StateHash]:
        return self._store.keys()

    def get(self, state_hash: StateHash) -> MCTSNodePy:
        return self._store[state_hash]

    def set(self, state_hash: StateHash, node: MCTSNodePy) -> None:
        self._store[state_hash] = node

    def clear(self) -> None:
        return self._store.clear()

    def load_from(self, node_store: NodeStore) -> None:
        return self._store.update({k: node_store.get(k) for k in node_store.keys()})  # noqa
