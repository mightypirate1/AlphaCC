from alpha_cc.api.game_manager.db import DB, DBGameState


class FauxDB(DB):
    def __init__(self) -> None:
        self._game_store: dict[str, DBGameState] = {}

    def get_entry(self, game_id: str) -> DBGameState:
        return self._game_store[game_id]

    def list_entries(self) -> list[str]:
        return list(self._game_store.keys())

    def set_entry(self, game_id: str, state: DBGameState) -> None:
        self._game_store[game_id] = state

    def remove_entry(self, game_id: str) -> bool:
        if success := game_id in self._game_store:
            del self._game_store[game_id]
        return success
