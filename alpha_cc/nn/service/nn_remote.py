import numpy as np
import torch

from alpha_cc.db.prediction_db import PredictionDB
from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.nn.nets.dual_head_net import DualHeadEvaluator
from alpha_cc.state import GameState


class NNRemote(DualHeadEvaluator):
    def __init__(self, pred_db: PredictionDB) -> None:
        self._pred_db = pred_db

    def get_pred(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        pi_tensor_unsoftmaxed, v_hat = self._get_pred(state)
        pi_vec_unsoftmaxed = pi_tensor_unsoftmaxed[*action_indexer(state)]
        pi = torch.nn.functional.softmax(pi_vec_unsoftmaxed, dim=0)
        return pi, v_hat

    def policy(self, state: GameState) -> np.ndarray:
        pi, _ = self.get_pred(state)
        return pi.numpy()

    def value(self, state: GameState) -> float:
        _, v_tensor = self.get_pred(state)
        return v_tensor.item()

    def clear_cache(self) -> None:
        pass  # this is handled by the service

    def _get_pred(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pred_db.has_pred(state):
            return self._pred_db.fetch_pred(state)
        self._pred_db.order_pred(state)
        return self._pred_db.await_pred(state)
