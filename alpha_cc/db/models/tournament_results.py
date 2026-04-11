from __future__ import annotations

from dataclasses import dataclass, field

from alpha_cc.db.models.game_result import GameResult


@dataclass
class _PairingStats:
    wins_p1: int = 0
    wins_p2: int = 0
    draws: int = 0
    timeouts: int = 0
    game_lengths: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins_p1 + self.wins_p2 + self.draws

    @property
    def p1_win_rate(self) -> float:
        return self.wins_p1 / max(self.total, 1)


class TournamentResult:
    """Aggregated tournament results built from per-game GameResult records.

    Usage:
    >>> results = TournamentResult.from_game_results(game_results)
    >>> results[1, 2]  # p1 win rate when channel 1 is p1 and channel 2 is p2
    """

    def __init__(self, pairings: dict[tuple[int, int], _PairingStats]) -> None:
        self._pairings = pairings

    @classmethod
    def from_game_results(cls, results: list[GameResult]) -> TournamentResult:
        pairings: dict[tuple[int, int], _PairingStats] = {}
        for r in results:
            key = (r.channel_p1, r.channel_p2)
            stats = pairings.setdefault(key, _PairingStats())
            stats.game_lengths.append(r.game_length)
            if r.hit_max_duration:
                stats.timeouts += 1
            if r.winner == 1:
                stats.wins_p1 += 1
            elif r.winner == 2:
                stats.wins_p2 += 1
            else:
                stats.draws += 1
        return cls(pairings)

    def __getitem__(self, key: tuple[int, int]) -> float:
        """Returns p1 win rate for the (p1_channel, p2_channel) pairing."""
        stats = self._pairings.get(key)
        if stats is None:
            return 0.0
        return stats.p1_win_rate

    @property
    def all_game_lengths(self) -> list[int]:
        return [gl for s in self._pairings.values() for gl in s.game_lengths]

    @property
    def total_games(self) -> int:
        return sum(s.total for s in self._pairings.values())

    @property
    def total_draws(self) -> int:
        return sum(s.draws for s in self._pairings.values())

    @property
    def total_timeouts(self) -> int:
        return sum(s.timeouts for s in self._pairings.values())
