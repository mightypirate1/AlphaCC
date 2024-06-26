import dill
import numpy as np
from redis import Redis

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store.node_store import NodeStore
from alpha_cc.engine import Board


class DBNodeStore(NodeStore):
    def __init__(self, game_id: str, db: Redis) -> None:
        self._game_id = game_id
        self._db = db

    def __contains__(self, board: Board) -> bool:
        return bool(self._db.hexists(self.nodes_key, hash(board)))  # type: ignore  # Board is bytes

    @property
    def nodes_key(self) -> str:
        return f"games-db/nodestore-nodes/{self._game_id}"

    @property
    def boards_key(self) -> str:
        return f"games-db/nodestore-boards/{self._game_id}"

    @property
    def updated_key(self) -> str:
        return f"games-db/nodestore-updated/{self._game_id}"

    def keys(self) -> list[Board]:
        board_hashes = self._db.hkeys(self.nodes_key)
        return [dill.loads(self._db.hget(self.boards_key, board_hash)) for board_hash in board_hashes]  # type: ignore  # noqa: S301

    def get(self, board: Board) -> MCTSNodePy:
        encoded = self._db.hget(self.nodes_key, hash(board))  # type: ignore
        if encoded is None:
            return MCTSNodePy(
                pi=np.zeros(0, dtype=np.float32),
                n=np.zeros(0, dtype=np.int32),
                q=np.zeros(0, dtype=np.float32),
                v_hat=0.0,
            )
        return dill.loads(encoded)  # noqa: S301

    def set(self, board: Board, node: MCTSNodePy) -> None:
        encoded_board = dill.dumps(board)
        encoded_node = dill.dumps(node)
        self._db.hset(self.boards_key, hash(board), encoded_board)  # type: ignore
        self._db.hset(self.nodes_key, hash(board), encoded_node)  # type: ignore
        self._db.lpush(self.updated_key, hash(board))

    def clear(self) -> None:
        self._db.delete(self.boards_key)
        self._db.delete(self.nodes_key)
        self._db.delete(self.updated_key)

    def fetch_updated(self, clear: bool = True) -> dict[Board, MCTSNodePy]:
        updated_board_hashes = self._db.lrange(self.updated_key, 0, -1)
        if updated_board_hashes is None:
            return {}
        if clear:
            self._db.delete(self.updated_key)
        updated_encoded_boards = self._db.hmget(self.boards_key, updated_board_hashes)  # type: ignore
        updated_encoded_nodes = self._db.hmget(self.nodes_key, updated_board_hashes)  # type: ignore
        updated_boards = [
            dill.loads(updated_encoded_board)  # noqa: S301
            for updated_encoded_board in updated_encoded_boards  # type: ignore
        ]
        updated_nodes = [
            dill.loads(updated_encoded_node)  # noqa: S301
            for updated_encoded_node in updated_encoded_nodes  # type: ignore
        ]
        return dict(zip(updated_boards, updated_nodes))

    def load_from(self, node_store: NodeStore | dict[Board, MCTSNodePy]) -> None:
        for board in node_store.keys():  # noqa
            self.set(board, node_store.get(board))  # type: ignore
