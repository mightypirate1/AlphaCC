from typing import Self

import numpy as np

_INT_INFINITY = 1000_0000_000_000  # infinity :-)


class ParamSchedule:
    """
    f(x) = clamp(
        (init + step * x [+ random]) * decay ^ x,
        min,
        max,
    )
    """

    def __init__(
        self,
        init: float | int,
        step: float = 0.0,
        decay: float = 1.0,
        random_add: float | None = None,
        min: float = -float("inf"),  # noqa: A002
        max: float = float("inf"),  # noqa: A002
    ) -> None:
        self._init = init
        self._step = step
        self._decay = decay
        self._random_add = random_add
        self._min = min
        self._max = max

    def as_float(self, t: float) -> float:
        x = self._init + t * self._step
        if self._random_add is not None:
            x += self._random_add * np.random.random()
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
