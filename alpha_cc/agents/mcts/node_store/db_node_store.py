import dill
from redis import Redis

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store.node_store import NodeStore
from alpha_cc.state.game_state import StateHash


class DBNodeStore(NodeStore):
    def __init__(self, game_id: str, db: Redis) -> None:
        self._game_id = game_id
        self._db = db

    def __contains__(self, state_hash: StateHash) -> bool:
        return bool(self._db.hexists(self.nodes_key, state_hash))  # type: ignore  # StateHash is bytes

    def keys(self) -> list[StateHash]:
        return self._db.hkeys(self.nodes_key)  # type: ignore

    @property
    def nodes_key(self) -> str:
        return f"games-db/nodes-cache/{self._game_id}"

    def get(self, state_hash: StateHash) -> MCTSNodePy:
        encoded = self._db.hget(self.nodes_key, state_hash)  # type: ignore  # StateHash is bytes
        if encoded is None:
            raise ValueError("node not found")
        return dill.loads(encoded)  # noqa: S301

    def set(self, state_hash: StateHash, node: MCTSNodePy) -> None:
        encoded = dill.dumps(node)
        self._db.hset(self.nodes_key, state_hash, encoded)  # type: ignore  # StateHash is bytes

    def clear(self) -> None:
        self._db.delete(self.nodes_key)

    def load_from(self, node_store: NodeStore) -> None:
        for state_hash in node_store.keys():  # noqa
            self.set(state_hash, node_store.get(state_hash))
