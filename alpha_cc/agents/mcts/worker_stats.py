from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerStats:
    """Per-game statistics from the MCTS worker's NN fetch loop."""

    total_fetch_time_us: int
    total_fetches: int

    @staticmethod
    def empty() -> WorkerStats:
        return WorkerStats(
            total_fetch_time_us=0,
            total_fetches=0,
        )

    @staticmethod
    def from_fetch_stats(fs: object) -> WorkerStats:
        """Construct from a Rust FetchStats pyclass instance."""
        return WorkerStats(
            total_fetch_time_us=fs.total_fetch_time_us,  # type: ignore[attr-defined]
            total_fetches=fs.total_fetches,  # type: ignore[attr-defined]
        )
