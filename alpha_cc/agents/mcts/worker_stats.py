from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerStats:
    """Per-game statistics from the MCTS worker's NN fetch loop.

    resolved_at_attempt[k] is the number of fetch_pred calls that resolved
    at attempt k (0 = cache hit, 1 = after first patience sleep, 2+ = backoff).
    attempt_total_wait_us[k] is the sum of actual elapsed time (microseconds)
    for all fetches that resolved at attempt k.
    """

    resolved_at_attempt: list[int]
    attempt_total_wait_us: list[int]
    timeouts: int
    total_gets: int
    total_misses: int
    total_fetch_time_us: int
    total_fetches: int
    current_patience_us: int

    @staticmethod
    def empty() -> WorkerStats:
        return WorkerStats(
            resolved_at_attempt=[],
            attempt_total_wait_us=[],
            timeouts=0,
            total_gets=0,
            total_misses=0,
            total_fetch_time_us=0,
            total_fetches=0,
            current_patience_us=0,
        )

    @staticmethod
    def from_fetch_stats(fs: object) -> WorkerStats:
        """Construct from a Rust FetchStats pyclass instance."""
        return WorkerStats(
            resolved_at_attempt=list(fs.resolved_at_attempt),  # type: ignore[attr-defined]
            attempt_total_wait_us=list(fs.attempt_total_wait_us),  # type: ignore[attr-defined]
            timeouts=fs.timeouts,  # type: ignore[attr-defined]
            total_gets=fs.total_gets,  # type: ignore[attr-defined]
            total_misses=fs.total_misses,  # type: ignore[attr-defined]
            total_fetch_time_us=fs.total_fetch_time_us,  # type: ignore[attr-defined]
            total_fetches=fs.total_fetches,  # type: ignore[attr-defined]
            current_patience_us=fs.current_patience_us,  # type: ignore[attr-defined]
        )
