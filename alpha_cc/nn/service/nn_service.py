import logging
from typing import Any

import torch
from apscheduler.schedulers.background import BackgroundScheduler

from alpha_cc.db import TrainingDB
from alpha_cc.engine import Board, NNPred, PredDB
from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.state import GameState

logger = logging.getLogger(__file__)
logging.getLogger("apscheduler").setLevel(logging.WARN)
scheduler = BackgroundScheduler()


class NNService:
    def __init__(
        self,
        nn: torch.nn.Module,
        pred_db: PredDB,
        training_db: TrainingDB,
        reload_frequency: int,
        log_frequency: int = 60,
    ) -> None:
        self._nn = nn
        self._pred_db = pred_db
        self._training_db = training_db
        self._nn.eval()
        self._current_weights_index = 0
        self._n_preds = 0
        self._n_batches = 0
        self._pred_db.flush_preds()
        self._initialize_scheduler(reload_frequency, log_frequency)

    def run(self) -> None:
        logger.info("Starting NNService")
        while True:
            boards = self._pred_db.fetch_all()
            self._process_request(boards)

    def update_weights(self, weights: dict[str, Any]) -> None:
        self._nn.load_state_dict(weights)
        self._nn.eval()
        self._pred_db.flush_preds()

    def _process_request(self, boards: list[Board]) -> None:
        states = [GameState(board) for board in boards]
        x = self._prepare_input(states)
        with torch.no_grad():
            x_pis, x_vals = self._nn(x)
        self._post_predictions(states, x_pis, x_vals)
        self._n_preds += len(boards)
        self._n_batches += 1

    def _prepare_input(self, states: list[GameState]) -> torch.Tensor:
        return torch.stack([state.tensor for state in states], dim=0)

    def _post_predictions(self, states: list[GameState], x_pis: torch.Tensor, x_vals: torch.Tensor) -> None:
        for state, pi_tensor_unsoftmaxed, val in zip(states, x_pis, x_vals):
            pi_unsoftmaxed = pi_tensor_unsoftmaxed[*action_indexer(state)]
            pi = torch.nn.functional.softmax(pi_unsoftmaxed, dim=0)
            nn_pred = NNPred(pi.numpy().tolist(), val.item())
            self._pred_db.post_pred(state.board, nn_pred)

    def _initialize_scheduler(self, reload_frequency: int, log_frequency: int) -> None:
        def reload_weights() -> None:
            if self._training_db.weights_is_latest(self._current_weights_index):
                return
            logger.info("Reloading weights")
            new_weights_index, weights = self._training_db.fetch_latest_weights_with_index()
            self.update_weights(weights)
            logger.info(f"Updated weights from {self._current_weights_index} to {new_weights_index}")
            self._current_weights_index = new_weights_index

        def log_stats() -> None:
            logger.info(f"Processed {self._n_preds} predictions in {self._n_batches} batches")
            self._n_preds = 0
            self._n_batches = 0

        scheduler.add_job(log_stats, "interval", seconds=log_frequency)
        scheduler.add_job(reload_weights, "interval", seconds=reload_frequency)
        scheduler.start()
