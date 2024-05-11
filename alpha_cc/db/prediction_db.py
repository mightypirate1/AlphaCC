import logging
from time import perf_counter_ns, sleep

import dill
import redis
import torch

from alpha_cc.state import GameState

logger = logging.getLogger(__file__)


class PredictionDB:
    def __init__(self, host: str = "localhost") -> None:
        self._queue_db = redis.Redis(host=host, db=1)
        self._pred_db = redis.Redis(host=host, db=2)

    @property
    def pred_key(self) -> str:
        return "prediction-queue"

    def post_pred(self, state: GameState, pi: torch.Tensor, value: torch.Tensor) -> None:
        self._pred_db.set(state.hash, dill.dumps((pi, value)))

    def await_pred(self, state: GameState, timeout: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        start_time = perf_counter_ns()
        while (pred := self._pred_db.get(state.hash)) is None:
            sleep(0.001)
            if start_time + timeout > perf_counter_ns():
                raise ValueError("timeout fetching pred")
        return dill.loads(pred)  # noqa

    def fetch_pred(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        return dill.loads(self._pred_db.get(state.hash))  # noqa

    def has_pred(self, state: GameState) -> bool:
        if self._pred_db.get(state.hash) is None:
            return False
        return True

    def flush_preds(self) -> None:
        logger.debug("clearing pred_db")
        self._pred_db.flushdb()

    def order_pred(self, state: GameState) -> None:
        self._queue_db.lpush(self.pred_key, dill.dumps(state))

    def fetch_all_states(self) -> list[GameState]:
        states = [self._fetch_state_blocking()]
        while (state := self._fetch_state()) is not None:
            states.append(state)
        return states

    def _fetch_state_blocking(self) -> GameState:
        _, encoded_state = self._queue_db.brpop(self.pred_key, timeout=0)  # type: ignore
        state = dill.loads(encoded_state)  # noqa
        return state

    def _fetch_state(self) -> GameState | None:
        encoded_state = self._queue_db.rpop(self.pred_key)
        if encoded_state is None:
            return None
        state = dill.loads(encoded_state)  # noqa
        return state
