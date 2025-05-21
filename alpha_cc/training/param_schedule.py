from typing import Self

_INT_INFINITY = 1000_0000_000_000  # infinity :-)


class ParamSchedule:
    def __init__(
        self,
        init: float | int,
        step: float = 0.0,
        decay: float = 1.0,
        min: float = -float("inf"),  # noqa: A002
        max: float = float("inf"),  # noqa: A002
    ) -> None:
        self._min = min
        self._max = max
        self._init = init
        self._step = step
        self._decay = decay

    def as_float(self, t: float) -> float:
        x = self._init + t * self._step
        x *= self._decay**t
        x = max(self._min, min(self._max, x))
        return x

    def as_int(self, t: float) -> int:
        x = self.as_float(t)
        if x == float("inf"):
            return _INT_INFINITY
        if x == -float("inf"):
            return -_INT_INFINITY
        return round(x)

    @classmethod
    def from_str(cls: type[Self], schedule_str: str) -> Self:
        def parse_entry(entry: str) -> tuple[str, int | float | bool]:
            kwarg, val = entry.split("=")
            try:
                return kwarg, float(val)
            except ValueError:
                raise ValueError(f"Could not convert {val} to float") from None

        # Parse as single litteral if possible
        try:
            init = float(schedule_str)
            return cls(init)
        except ValueError:
            pass

        # Parse as a list of keyword arguments
        kwargs = dict(parse_entry(entry) for entry in schedule_str.replace(" ", "").split(","))
        return cls(**kwargs)
