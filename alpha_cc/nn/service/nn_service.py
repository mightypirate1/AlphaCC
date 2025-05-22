import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import torch
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler

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
        nn_creator: Callable[[], torch.nn.Module],
        nn: torch.nn.Module,
        current_weights_index: int,
        inference_batch_size: int = 512,
        fetch_batch_size: int = 2048,
        num_post_workers: int = 2,
        log_frequency: int = 60,
        device: str = "cpu",
    ) -> None:
        self._pred_db_channel = pred_db_channel
        self._training_db = training_db
        self._nn_creator = nn_creator
        self._current_weights_index = current_weights_index
        self._inference_batch_size = inference_batch_size
        self._fetch_batch_size = fetch_batch_size
        self._log_frequency = log_frequency
        self._device = torch.device(device)
        self._n_preds = 0
        self._n_batches = 0
        self._nn_prediction = self._configure_nn(nn)  # served nn
        self._nn_loading = self._configure_nn(nn)  # used for loading weights
        self._jobs = self._initialize_scheduler()
        self._post_pool = ThreadPoolExecutor(max_workers=num_post_workers)
        self._is_loading_weights = False
        pred_db_channel.flush_preds()

    @property
    def channel(self) -> int:
        return self._pred_db_channel.channel

    @property
    def current_weights_index(self) -> int:
        return self._current_weights_index

    def process_requests(self) -> None:
        boards = self._pred_db_channel.fetch_requests(self._fetch_batch_size)
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
                x_batch_pis, x_batch_vals = self._nn_prediction(x_batch)

            # post in a separate thread to avoid blocking GPU
            self._post_pool.submit(
                self._post_predictions,
                batch_states,
                x_batch_pis.detach().cpu(),
                x_batch_vals.detach().cpu(),
            )

        self._n_preds += len(states)
        self._n_batches += 1

    def deactivate(self) -> None:
        for job in self._jobs:
            job.remove()

    def load_weights(self, target_weight_index: int) -> None:
        if target_weight_index == self._current_weights_index or self._is_loading_weights:
            return

        self._is_loading_weights = True

        def background_loading() -> None:
            try:
                # get weights from the training db into the loading nn
                weights = self._training_db.weights_fetch(target_weight_index)
                self._nn_loading.load_state_dict(weights)
                # swap the loading nn with the prediction nn
                old_model = self._nn_prediction
                self._nn_prediction = self._nn_loading
                self._nn_loading = old_model

                self._pred_db_channel.flush_preds()
                logger.info(
                    f"[Channel-{self._pred_db_channel.channel}]:"
                    f" updated weights {self._current_weights_index}->{target_weight_index}"
                )
                self._current_weights_index = target_weight_index
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to load weights: {e}")
            finally:
                self._is_loading_weights = False

        Thread(target=background_loading).start()

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

    def _configure_nn(self, nn: torch.nn.Module) -> torch.nn.Module:
        # TODO: make jit-compilation work
        # nn = torch.jit.script(nn)
        nn = nn.to(self._device)
        nn = nn.eval()
        return nn

    def _initialize_scheduler(self) -> list[Job]:
        stats_job = scheduler.add_job(self._log_stats, "interval", seconds=self._log_frequency)
        return [stats_job]

    def _log_stats(self) -> None:
        logger.info(
            f"[Channel-{self._pred_db_channel.channel}]: Processed"
            f" {self._n_preds} predictions in {self._n_batches} batches"
        )
        self._n_preds = 0
        self._n_batches = 0


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
        pred_db_channel = PredDBChannel(self._redis_host_pred, channel)
        nn = self._nn_creator()
        served_nn = ServedNN(
            pred_db_channel,
            self._training_db,
            self._nn_creator,
            nn,
            weight_index,
            inference_batch_size=self._infecence_batch_size,
            num_post_workers=self._num_post_workers,
            log_frequency=self._log_frequency,
            device=self._device,
        )
        served_nn.load_weights(weight_index)
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
