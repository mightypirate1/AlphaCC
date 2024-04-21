from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, ParamSpec, TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


@dataclass
class DbgStats:
    tag: str
    start_time: float
    stop_time: float | None = None

    @property
    def duration(self) -> float:
        if self.stop_time is None:
            raise ValueError("wait for completion before calling me")
        return self.stop_time - self.start_time


class Timer:
    def __init__(self, tag: str, verbose: bool = False) -> None:
        self._tag = tag
        self._verbose = verbose
        self._dbg_stats: DbgStats
        self._start_time: float
        self._stop_time: float

    def __enter__(self) -> DbgStats:
        self._start_time = perf_counter()
        self._dbg_stats = DbgStats(tag=self._tag, start_time=self._start_time)
        return self._dbg_stats

    def __exit__(self, *_: Any, **__: Any) -> None:
        self._stop_time = perf_counter()
        self._dbg_stats.stop_time = self._stop_time
        if self._verbose:
            print(f"{self._tag} ran for {self._dbg_stats.duration} seconds")  # noqa


class Profiler:
    def __init__(self) -> None:
        self._stats: dict[str, float] = {}

    def __call__(self, func: Callable[Param, RetType]) -> Callable[Param, RetType]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tag = func.__qualname__
            with Timer(tag) as stats:
                ret = func(*args, **kwargs)
            self._stats[tag] = self._stats.get(tag, 0.0) + stats.duration
            return ret

        return wrapper

    @property
    def stats(self) -> dict[str, float]:
        return self._stats
