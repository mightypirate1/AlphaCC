from typing import Self

from redis import Redis


class TournamentResult:
    """
    Used by TrainingDB to report tournament results.

    Usage:
    >>> results = db.tournament_get_results()
    >>> results[1, 2]  # win rate of channel 1 (as white) against channel 2 (as black)
    >>> results[2, 1]  # win rate of channel 1 (as black) against channel 2 (as white)

    """

    def __init__(self, wins: dict[int, dict[int, int]], counts: dict[int, dict[int, int]]) -> None:
        self._wins = wins
        self._counts = counts

    def __getitem__(self, key: tuple[int, int]) -> float:
        player_1, player_2 = key
        wins = self._wins.get(player_1, {}).get(player_2, 0.0)
        count = self._counts.get(player_1, {}).get(player_2, 1.0)
        return wins / count

    @classmethod
    def from_db(cls: type[Self], tournament_result_key: str, db: Redis) -> Self:
        def increment(x: dict[int, dict[int, int]], player_1: int, player_2: int, value: int) -> None:
            if player_1 not in x:
                x[player_1] = {}
            x[player_1][player_2] = x[player_1].get(player_2, 0) + value

        data = db.hgetall(tournament_result_key)
        wins: dict[int, dict[int, int]] = {}
        counts: dict[int, dict[int, int]] = {}
        for key, value in data.items():  # type: ignore
            player_1, player_2, winner = map(int, key.decode().split("-"))
            increment(counts, player_1, player_2, int(value))
            if winner == player_1:
                increment(wins, player_1, player_2, int(value))
        return cls(wins, counts)
