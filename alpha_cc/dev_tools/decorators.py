from collections.abc import Callable
from typing import Any, TypeVar

from alpha_cc.dev_tools.timer import Timer

F = TypeVar("F", bound=Callable[..., Any])


def dbg_timer(func: F) -> F:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(func.__name__, verbose=True):
            ret = func(*args, **kwargs)
        return ret

    return wrapper  # type: ignore  # i tried :-(
