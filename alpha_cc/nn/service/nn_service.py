import logging
from collections.abc import Callable

import torch
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler

from concurrent.futures import ThreadPoolExecutor

from alpha_cc.db import TrainingDB
from alpha_cc.engine import NNPred, PredDBChannel
from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.state import GameState

logger = logging.getLogger(__file__)
logging.getLogger("apscheduler").setLevel(logging.WARN)
scheduler = BackgroundScheduler()


class ServedNN:
    def __init__(
        self,
        pred_db_channel: PredDBChannel,
        training_db: TrainingDB,
        nn: torch.nn.Module,
        inference_batch_size: int = 512,
        num_post_workers: int = 2,
        log_frequency: int = 60,
        reload_frequency: int = 1,
        device: str = "cpu",
    ) -> None:
        self._pred_db_channel = pred_db_channel
        self._training_db = training_db
        self._inference_batch_size = inference_batch_size
        self._log_frequency = log_frequency
        self._reload_frequency = reload_frequency
        self._device = torch.device(device)
        self._nn: torch.nn.Module | None = None
        self._current_weights_index = -1
        self._n_preds = 0
        self._n_batches = 0
        self._jobs = self._initialize_scheduler()
        self._assign_nn(nn)
        self._post_pool = ThreadPoolExecutor(max_workers=num_post_workers)
        pred_db_channel.flush_preds()

    @property
    def channel(self) -> int:
        return self._pred_db_channel.channel

    def process_requests(self) -> None:
        if self._nn is None:
            raise ValueError(f"can not call a deactivated ServedNN[channel={self.channel}]")
        boards = self._pred_db_channel.fetch_all()
        if len(boards) == 0:
            return
        states = [GameState(board) for board in boards]
        x = self._prepare_input(states)
        
        # batch it up, in case we have a LOT of requests
        for i in range(0, len(states), self._inference_batch_size):
            batch_states = states[i : i + self._inference_batch_size]
            x_batch = x[i : i + self._inference_batch_size]

            if len(batch_states) == 0:
                continue

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device == "cuda"):
                x_batch_pis, x_batch_vals = self._nn(x_batch)

            # post in a separate thread to avoid blocking GPU
            self._post_pool.submit(
                self._post_predictions,
                    batch_states,
                    x_batch_pis.detach().to("cpu", non_blocking=True),
                    x_batch_vals.detach().to("cpu", non_blocking=True),
                )

        self._n_preds += len(states)
        self._n_batches += 1

    def deactivate(self) -> None:
        self._nn = None
        for job in self._jobs:
            job.remove()

    def activate(self, nn: torch.nn.Module) -> None:
        self._nn = nn
        self._pred_db_channel.flush_preds()
        self._jobs = self._initialize_scheduler()

    def _prepare_input(self, states: list[GameState]) -> torch.Tensor:
        """assumes at least one state is passed"""
        batch_size = len(states)
        state_shape = states[0].tensor.shape
        input_tensor = torch.empty(batch_size, *state_shape, pin_memory=True)
        for i, state in enumerate(states):
            input_tensor[i] = state.tensor
        return input_tensor.to(self._device, non_blocking=True)

    def _post_predictions(self, states: list[GameState], x_pis: torch.Tensor, x_vals: torch.Tensor) -> None:
        """
        Post predictions to the prediction database.

        They are assumed to be on the CPU, and in the same order as the states.
        """
        def create_nn_pred(state: GameState, pi_tensor_unsoftmaxed: torch.Tensor, val: torch.Tensor) -> NNPred:
            pi_unsoftmaxed = pi_tensor_unsoftmaxed[*action_indexer(state)]
            pi = torch.nn.functional.softmax(pi_unsoftmaxed, dim=0)
            return NNPred(pi.numpy().tolist(), val.item())

        nn_preds = [
            create_nn_pred(state, pi_tensor_unsoftmaxed, val)
            for state, pi_tensor_unsoftmaxed, val in zip(states, x_pis, x_vals)
        ]
        for state, nn_pred in zip(states, nn_preds):
            self._pred_db_channel.post_pred(state.board, nn_pred)

    def _assign_nn(self, nn: torch.nn.Module) -> None:
        self._nn = nn
        # TODO: make jit-compilation work
        # self._nn = torch.jit.script(nn)
        self._nn.to(self._device)
        self._nn.eval()

    def _initialize_scheduler(self) -> list[Job]:
        stats_job = scheduler.add_job(self._log_stats, "interval", seconds=self._log_frequency)
        weights_job = scheduler.add_job(self._update_weights, "interval", seconds=self._reload_frequency)
        return [stats_job, weights_job]

    def _log_stats(self) -> None:
        logger.info(
            f"[Channel-{self._pred_db_channel.channel}]: Processed"
            f" {self._n_preds} predictions in {self._n_batches} batches"
        )
        self._n_preds = 0
        self._n_batches = 0

    def _update_weights(self) -> None:
        if self._nn is None:
            logger.warning(f"[Channel-{self._pred_db_channel.channel}]: attempted reload but is deactivated")
            return
        target_weight_index = self._training_db.model_get_current().get(self.channel)
        if target_weight_index is None:
            logger.warning(f"[Channel-{self._pred_db_channel.channel}]: attempted reload but has no target weights")
            return

        if target_weight_index == self._current_weights_index:
            return

        self._current_weights_index = target_weight_index
        weights = self._training_db.weights_fetch(target_weight_index)
        self._nn.load_state_dict(weights)
        self._nn.eval()
        self._pred_db_channel.flush_preds()
        logger.info(f"[Channel-{self._pred_db_channel.channel}]: updated weights to {target_weight_index}")


class NNService:
    def __init__(
        self,
        nn_creator: Callable[[], torch.nn.Module],
        redis_host_main: str = "localhost",
        redis_host_pred: str = "localhost",
        log_frequency: int = 60,
        reload_frequency: int = 1,
        infecence_batch_size: int = 512,
        num_post_workers: int = 2,
        gpu: bool = False,
    ) -> None:
        self._nn_creator = nn_creator
        self._redis_host_pred = redis_host_pred
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
        while True:
            for served_nn in self._served_nns:
                served_nn.process_requests()

    def add_channel(self, channel: int) -> None:
        for served_nn in self._served_nns:
            if served_nn.channel == channel:
                logger.warning(f"[Channel-{channel}]: already exists")
                return
        self._served_nns.append(self._create_served_nn(channel))

    def drop_channel(self, channel: int) -> None:
        for served_nn in self._served_nns:
            if served_nn.channel == channel:
                served_nn.deactivate()
                self._served_nns.remove(served_nn)
                logger.info(f"[Channel-{channel}]: dropped")
                return

    def has_channel(self, channel: int) -> bool:
        return any(snn.channel == channel for snn in self._served_nns)

    def _create_served_nn(self, channel: int) -> ServedNN:
        pred_db_channel = PredDBChannel(self._redis_host_pred, channel)
        nn = self._nn_creator()
        return ServedNN(
            pred_db_channel,
            self._training_db,
            nn,
            inference_batch_size=self._infecence_batch_size,
            num_post_workers=self._num_post_workers,
            log_frequency=self._log_frequency,
            reload_frequency=self._reload_frequency,
            device=self._device,
        )

    def _initialize_scheduler(self) -> None:
        def update_served_nets() -> None:
            current_channels = self._training_db.model_get_current().keys()
            for channel in current_channels:
                if not self.has_channel(channel):
                    self.add_channel(channel)
            for served_nn in self._served_nns:
                if served_nn.channel not in current_channels:
                    self.drop_channel(served_nn.channel)

        scheduler.add_job(update_served_nets, "interval", seconds=self._reload_frequency)
        scheduler.start()
