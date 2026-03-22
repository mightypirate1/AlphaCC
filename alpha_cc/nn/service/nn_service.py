import logging
import signal
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, Thread
from typing import Any

import dill
import torch
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler

from alpha_cc.db import TrainingDB
from alpha_cc.engine import (
    InferenceBatch,
    enqueue_responses,
    fetch_and_build_tensor,
)

logger = logging.getLogger(__file__)
logging.getLogger("apscheduler").setLevel(logging.WARN)
scheduler = BackgroundScheduler()
stop_signal = threading.Event()


def _signal_handler(signum: int, _: Any) -> None:
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    scheduler.shutdown(wait=False)
    stop_signal.set()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class ServedNN:
    def __init__(
        self,
        channel: int,
        training_db: TrainingDB,
        nn_creator: Callable[[], torch.nn.Module],
        nn: torch.nn.Module,
        current_weights_index: int,
        board_size: int = 9,
        inference_batch_size: int = 512,
        num_post_workers: int = 2,
        prefetch_size: int = 5,
        log_frequency: int = 60,
        device: str = "cpu",
    ) -> None:
        self._channel = channel
        self._training_db = training_db
        self._nn_creator = nn_creator
        self._current_weights_index = current_weights_index
        self._board_size = board_size
        self._inference_batch_size = inference_batch_size
        self._log_frequency = log_frequency
        self._device = torch.device(device)
        self._n_preds = 0
        self._n_batches = 0
        self._time_wait_s = 0.0
        self._time_inference_s = 0.0
        self._time_post_s = 0.0
        self._time_prefetch_s = 0.0
        self._nn_prediction = self._configure_nn(nn)  # served nn
        self._nn_loading = self._configure_nn(nn)  # used for loading weights
        self._jobs = self._initialize_scheduler()
        self._post_pool = ThreadPoolExecutor(max_workers=num_post_workers)
        self._is_loading_weights = False
        self._inference_buffer = torch.empty(inference_batch_size, 2, board_size, board_size, device=self._device)
        self._prefetch_batches: deque[InferenceBatch] = deque(maxlen=prefetch_size)
        self._prefetch_tensors: deque[torch.Tensor] = deque(maxlen=prefetch_size)
        self._prefetch_condition = Condition()
        self._prefetch_stop = Event()
        self._prefetch_threads: list[Thread] = []
        self._start_prefetch_thread()

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def current_weights_index(self) -> int:
        return self._current_weights_index

    def process_requests(self) -> None:
        t0 = time.monotonic()

        # Wait for at least one batch, then drain all available
        with self._prefetch_condition:
            while not self._prefetch_batches:
                if self._prefetch_stop.is_set() or stop_signal.is_set():
                    return
                self._prefetch_condition.wait(timeout=0.1)
            batches: list[InferenceBatch] = []
            tensors: list[torch.Tensor] = []
            while self._prefetch_batches:
                batches.append(self._prefetch_batches.popleft())
                tensors.append(self._prefetch_tensors.popleft())
            self._prefetch_condition.notify_all()

        t1 = time.monotonic()

        # Merge tensors into pre-allocated buffer for GPU efficiency, but keep
        # original batches intact — each batch carries its own slot for response routing.
        n_items = sum(len(b) for b in batches)
        buf_pos = 0
        src_idx = 0
        src_off = 0

        while src_idx < len(tensors) or buf_pos > 0:
            # Fill buffer up to inference_batch_size
            while buf_pos < self._inference_batch_size and src_idx < len(tensors):
                src = tensors[src_idx]
                avail = src.shape[0] - src_off
                space = self._inference_batch_size - buf_pos
                n = min(avail, space)
                self._inference_buffer[buf_pos : buf_pos + n].copy_(src[src_off : src_off + n])
                buf_pos += n
                src_off += n
                if src_off >= src.shape[0]:
                    src_idx += 1
                    src_off = 0

            if buf_pos == 0:
                break

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self._device.type == "cuda"):
                x_batch_pis, x_batch_vals = self._nn_prediction(self._inference_buffer[:buf_pos])

            # Split output back to original batches for correct per-slot routing
            out_off = 0
            while batches and out_off < buf_pos:
                batch = batches[0]
                batch_len = len(batch)
                take = min(batch_len, buf_pos - out_off)
                if take == batch_len:
                    # Whole batch fits in this inference chunk
                    batches.pop(0)
                    sub_batch = batch
                else:
                    # Batch is split across inference chunks — post the part that fits,
                    # keep the remainder for the next chunk
                    sub_batch = batch.slice(0, take)
                    batches[0] = batch.slice(take, batch_len)

                self._post_pool.submit(
                    self._post_predictions,
                    sub_batch,
                    x_batch_pis[out_off : out_off + take].detach().cpu(),
                    x_batch_vals[out_off : out_off + take].detach().cpu(),
                )
                out_off += take

            buf_pos = 0

        t2 = time.monotonic()

        self._time_wait_s += t1 - t0
        self._time_inference_s += t2 - t1
        self._n_preds += n_items
        self._n_batches += 1

    def deactivate(self) -> None:
        self._prefetch_stop.set()
        with self._prefetch_condition:
            self._prefetch_condition.notify_all()
        for t in self._prefetch_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._prefetch_threads.clear()
        for job in self._jobs:
            job.remove()

    def load_weights(self, target_weight_index: int, force: bool = False) -> None:
        """Load weights for the served NN.

        Parameters
        ----------
        target_weight_index: int
            The weight index to load from the training DB.
        force: bool
            If True, load even if target_weight_index == current_weights_index.
        """
        if (target_weight_index == self._current_weights_index and not force) or self._is_loading_weights:
            return

        self._is_loading_weights = True

        def background_loading() -> None:
            try:
                # get weights from the training db into the loading nn
                payload = self._training_db.weights_fetch(target_weight_index)
                weights = dill.loads(payload)  # noqa
                self._nn_loading.load_state_dict(weights)
                # swap the loading nn with the prediction nn
                old_model = self._nn_prediction
                self._nn_prediction = self._nn_loading
                self._nn_loading = old_model

                logger.info(
                    f"[Channel-{self._channel}]:"
                    f" updated weights {self._current_weights_index}->{target_weight_index}"
                )
                self._current_weights_index = target_weight_index
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to load weights: {e}")
            finally:
                self._is_loading_weights = False

        Thread(target=background_loading).start()

    def _post_predictions(self, batch: InferenceBatch, x_pis: torch.Tensor, x_vals: torch.Tensor) -> None:
        try:
            t0 = time.monotonic()
            enqueue_responses(
                x_pis.float().numpy().ravel(),
                x_vals.float().numpy().ravel(),
                batch,
            )
            self._time_post_s += time.monotonic() - t0
        except Exception:
            logger.exception("_post_predictions failed — shutting down")
            stop_signal.set()

    def _configure_nn(self, nn: torch.nn.Module) -> torch.nn.Module:
        # TODO: make jit-compilation work
        # nn = torch.jit.script(nn)
        nn = nn.to(self._device)
        nn = nn.eval()
        return nn

    def _start_prefetch_thread(self) -> None:
        """Start a thread that continuously prefetches board requests."""

        def prefetch_worker() -> None:
            try:
                while not self._prefetch_stop.is_set() and not stop_signal.is_set():
                    # Wait if queue is full
                    with self._prefetch_condition:
                        while len(self._prefetch_batches) >= (self._prefetch_batches.maxlen or 1000):
                            if self._prefetch_stop.is_set() or stop_signal.is_set():
                                break
                            self._prefetch_condition.wait(timeout=0.1)

                    if self._prefetch_stop.is_set() or stop_signal.is_set():
                        break

                    t0 = time.monotonic()
                    result = fetch_and_build_tensor(
                        self._channel,
                        self._inference_batch_size,
                        self._board_size,
                    )
                    if result is None:
                        time.sleep(0.0001)
                        continue

                    batch, arr = result
                    x = torch.from_numpy(arr).pin_memory().to(self._device, non_blocking=True)
                    self._time_prefetch_s += time.monotonic() - t0

                    with self._prefetch_condition:
                        self._prefetch_batches.append(batch)
                        self._prefetch_tensors.append(x)
                        self._prefetch_condition.notify()  # wake main thread
            except Exception:
                logger.exception(f"[Channel-{self._channel}]: Prefetch thread crashed — shutting down")
                stop_signal.set()

        thread = Thread(target=prefetch_worker, daemon=True)
        thread.start()
        self._prefetch_threads.append(thread)
        logger.info(f"[Channel-{self._channel}]: Started prefetch thread")

    def _initialize_scheduler(self) -> list[Job]:
        stats_job = scheduler.add_job(self._log_stats, "interval", seconds=self._log_frequency)
        return [stats_job]

    def _log_stats(self) -> None:
        prefetch_usage = 0
        with self._prefetch_condition:
            prefetch_usage = len(self._prefetch_batches)

        logger.info(
            f"[Channel-{self._channel}]: Processed"
            f" {self._n_preds} predictions in {self._n_batches} batches"
            f" (Prefetch: {prefetch_usage}/{self._prefetch_batches.maxlen})"
            f" | wait={self._time_wait_s:.2f}s"
            f" infer={self._time_inference_s:.2f}s"
            f" post={self._time_post_s:.2f}s"
            f" prefetch={self._time_prefetch_s:.2f}s"
        )

        self._n_preds = 0
        self._n_batches = 0
        self._time_wait_s = 0.0
        self._time_inference_s = 0.0
        self._time_post_s = 0.0
        self._time_prefetch_s = 0.0


class NNService:
    def __init__(
        self,
        nn_creator: Callable[[], torch.nn.Module],
        board_size: int = 9,
        redis_host_main: str = "localhost",
        log_frequency: int = 60,
        reload_frequency: int = 1,
        infecence_batch_size: int = 512,
        num_post_workers: int = 2,
        gpu: bool = False,
    ) -> None:
        self._nn_creator = nn_creator
        self._board_size = board_size
        self._log_frequency = log_frequency
        self._reload_frequency = reload_frequency
        self._infecence_batch_size = infecence_batch_size
        self._num_post_workers = num_post_workers
        self._device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self._training_db = TrainingDB(host=redis_host_main)
        self._n_preds = 0
        self._n_batches = 0
        self._served_nns: list[ServedNN] = []
        self._initialize_scheduler()

    def run(self) -> None:
        logger.info(f"Starting NNService[{self._device}]")
        while stop_signal is None or not stop_signal.is_set():
            for served_nn in self._served_nns:
                served_nn.process_requests()

    def add_channel(self, channel: int, weight_index: int) -> None:
        for served_nn in self._served_nns:
            if served_nn.channel == channel:
                logger.warning(f"[Channel-{channel}]: already exists")
                return
        self._served_nns.append(self._create_served_nn(channel, weight_index))

    def get_served_nn_by_channel(self, channel: int) -> ServedNN | None:
        for served_nn in self._served_nns:
            if served_nn.channel == channel:
                return served_nn
        logger.warning(f"[Channel-{channel}]: not found")
        return None

    def drop_channel(self, channel: int) -> None:
        for served_nn in self._served_nns:
            if served_nn.channel == channel:
                served_nn.deactivate()
                self._served_nns.remove(served_nn)
                logger.info(f"[Channel-{channel}]: dropped")
                return

    def has_channel(self, channel: int) -> bool:
        return any(snn.channel == channel for snn in self._served_nns)

    def _create_served_nn(self, channel: int, weight_index: int) -> ServedNN:
        nn = self._nn_creator()
        served_nn = ServedNN(
            channel,
            self._training_db,
            self._nn_creator,
            nn,
            weight_index,
            board_size=self._board_size,
            inference_batch_size=self._infecence_batch_size,
            num_post_workers=self._num_post_workers,
            log_frequency=self._log_frequency,
            device=self._device,
        )
        # Force initial load so that on restart the restored weights are actually served immediately
        served_nn.load_weights(weight_index, force=True)
        return served_nn

    def _initialize_scheduler(self) -> None:
        def update_served_nets() -> None:
            current_models = self._training_db.model_get_current()  # what the trainer is telling us to serve
            for channel, weight_idx in current_models.items():
                if not self.has_channel(channel):
                    self.add_channel(channel, weight_idx)
                else:
                    channel_nn = self.get_served_nn_by_channel(channel)
                    if channel_nn is not None:
                        channel_nn.load_weights(weight_idx)
                    else:
                        logger.warning(f"[Channel-{channel}]: not found in served nns")

            for served_nn in self._served_nns:
                if served_nn.channel not in current_models:
                    self.drop_channel(served_nn.channel)

        scheduler.add_job(update_served_nets, "interval", seconds=self._reload_frequency)
        scheduler.start()
