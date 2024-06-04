import dill
from redis import Redis

from alpha_cc.db.models import DBGameState


class GamesDB:
    def __init__(self, host: str = "localhost") -> None:
        self._db = Redis(host=host, db=3)

    @property
    def games_key(self) -> str:
        return "games-db/games-cache"

    def get_entry(self, game_id: str) -> DBGameState:
        encoded = self._db.hget(self.games_key, game_id)
        if encoded is None:
            raise ValueError(f"No game with id={game_id} found")
        return dill.loads(encoded)  # noqa

    def list_entries(self) -> list[str]:
        keys = self._db.hkeys(self.games_key)
        return [k.decode() for k in keys]  # type: ignore

    def set_entry(self, game_id: str, state: DBGameState) -> None:
        encoded = dill.dumps(state)
        self._db.hset(self.games_key, game_id, encoded)

    def remove_entry(self, game_id: str) -> bool:
        existed = bool(self._db.hexists(self.games_key, game_id))
        self._db.hdel(self.games_key, game_id)
        return existed
