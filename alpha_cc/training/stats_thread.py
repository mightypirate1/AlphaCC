import logging
import threading
from collections import deque
from typing import Any

from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.training.trainer import Trainer

logger = logging.getLogger(__name__)


class StatsThread(threading.Thread):
    """
    Background thread that runs report_rollout_stats on a separate Trainer instance.

    Uses a deque as a bounded work queue — if the thread falls behind, oldest items
    are dropped (we only care about the freshest stats). Adaptive limit shrinks the
    eval sample count proportionally to queue fill fraction so it catches up faster.
    """

    _QUEUE_MAXLEN = 10

    def __init__(self, trainer: Trainer, base_limit: int) -> None:
        super().__init__(daemon=True, name="stats-thread")
        self._trainer = trainer
        self._base_limit = base_limit
        self._queue: deque[tuple[dict[str, Any], list[TrainingData], int]] = deque(maxlen=self._QUEUE_MAXLEN)
        self._lock = threading.Lock()
        self._work_available = threading.Event()

    def submit(
        self,
        state_dict: dict[str, Any],
        training_datas: list[TrainingData],
        global_step: int,
    ) -> None:
        with self._lock:
            self._queue.append((state_dict, training_datas, global_step))
        self._work_available.set()

    def run(self) -> None:
        while True:
            self._work_available.wait()
            with self._lock:
                if not self._queue:
                    self._work_available.clear()
                    continue
                fill_fraction = len(self._queue) / self._QUEUE_MAXLEN
                state_dict, training_datas, global_step = self._queue.popleft()
                if not self._queue:
                    self._work_available.clear()

            effective_limit = max(1, int(self._base_limit * (1.0 - fill_fraction)))
            if fill_fraction > 0:
                logger.debug(f"stats-thread queue fill={fill_fraction:.0%}, " f"effective_limit={effective_limit}")

            try:
                self._trainer.nn.load_state_dict(state_dict)
                self._trainer.set_steps(global_step, global_step)
                self._trainer.report_rollout_stats(training_datas, limit=effective_limit)
            except Exception:
                logger.exception("stats-thread error")
