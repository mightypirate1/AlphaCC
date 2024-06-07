from redis import Redis

from alpha_cc.db.models import DBGameState


class GamesDB:
    def __init__(self, host: str = "localhost") -> None:
        self._db = Redis(host=host, db=3)

    @property
    def db(self) -> Redis:
        return self._db

    @property
    def games_key(self) -> str:
        return "games-db/games-list"

    def get_game_key(self, game_id: str) -> str:
        return f"games-db/game-moves/{game_id}"

    def create_game(self, game_id: str, size: int) -> DBGameState:
        self._db.hset(self.games_key, game_id, size)  # type: ignore
        return DBGameState(game_id, size, [])

    def get_state(self, game_id: str) -> DBGameState:
        size = self._get_size(game_id)
        if size is None:
            raise ValueError(f"game_id={game_id} does not exist")
        move_indices = []
        if self._db.exists(self.get_game_key(game_id)):
            encoded_move_indices = self._db.lrange(self.get_game_key(game_id), 0, -1)
            move_indices = [int(m) for m in encoded_move_indices]  # type: ignore
        return DBGameState(game_id, size, move_indices)

    def list_entries(self) -> list[str]:
        if self._db.exists(self.games_key):
            game_keys = self._db.hkeys(self.games_key)
            return [k.decode() for k in game_keys]  # type: ignore
        return []

    def add_move(self, game_id: str, move_index: int) -> None:
        self._db.rpush(self.get_game_key(game_id), move_index)

    def remove_entry(self, game_id: str) -> bool:
        game_key = self.get_game_key(game_id)
        existed = bool(self._db.exists(game_key))
        self._db.unlink(game_key)
        self._db.hdel(self.games_key, game_id)
        return existed

    def _get_size(self, game_id: str) -> int | None:
        encoded_game_size = self._db.hget(self.games_key, game_id)
        if encoded_game_size is None:
            return None
        return int(encoded_game_size)  # type: ignore
