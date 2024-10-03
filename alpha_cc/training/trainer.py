import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.nn.blocks import PolicyLogSoftmax, PolicySoftmax
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.training.training_dataset import TrainingDataset


class Trainer:
    def __init__(
        self,
        board_size: int,
        nn: DefaultNet,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.0,
        l2_reg: float = 1e-4,
        epochs_per_update: int = 3,
        batch_size: int = 64,
        lr: float = 1e-4,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._nn = nn
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight
        self._epochs_per_update = epochs_per_update
        self._batch_size = batch_size
        self._policy_log_softmax = PolicyLogSoftmax(board_size)
        self._policy_softmax = PolicySoftmax(board_size)
        self._optimizer = torch.optim.AdamW(nn.parameters(), lr=lr, weight_decay=l2_reg)
        self._global_step = 0
        self._eval_step = 0
        self._summary_writer = summary_writer

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    def train(self, dataset: TrainingDataset) -> None:
        self._update_nn(dataset)
        self._global_step += 1

    def set_lr(self, lr: float) -> None:
        for g in self._optimizer.param_groups:
            g["lr"] = lr

    def report_rollout_stats(self, trajectories: list[list[MCTSExperience]]) -> None:
        def log_aggregates(key: str, data: np.ndarray | torch.Tensor, prefix: str = "train-rollouts") -> None:
            if self._summary_writer is None:
                return
            self._summary_writer.add_scalar(f"{prefix}/{key}-mean", data.mean(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"{prefix}/{key}-min", data.min(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"{prefix}/{key}-max", data.max(), global_step=self._global_step)

        def entropy(x: torch.Tensor | np.ndarray) -> torch.Tensor:
            x = x.reshape(x.shape[0], -1)
            return -torch.sum(torch.as_tensor(x) * torch.log(torch.as_tensor(x.clip(1e-6))), dim=-1)

        if self._summary_writer is None:
            return

        game_lengths = np.array([len(traj) for traj in trajectories])
        v_targets = np.array([e.v_target for traj in trajectories for e in traj])
        pi_targets_logits_flat = np.concatenate(
            [np.log(e.pi_target.clip(1e-6).ravel()) for traj in trajectories for e in traj]
        )
        eval_dataset = TrainingDataset([exp for traj in trajectories for exp in traj])
        pi_logits, v = self._evaluate(eval_dataset)
        pi_targets = torch.stack([pi_target for _, _, pi_target, _, _ in eval_dataset])  # type: ignore
        pi_target_entropy = entropy(pi_targets)
        self._summary_writer.add_histogram(
            "trainer/pi-target-entropy", pi_target_entropy, global_step=self._global_step
        )
        self._summary_writer.add_histogram("trainer/pi-pred-logits", pi_logits, global_step=self._global_step)
        self._summary_writer.add_histogram("trainer/v-pred", v, global_step=self._global_step)
        self._summary_writer.add_histogram(
            "trainer/pi-target-logits", pi_targets_logits_flat, global_step=self._global_step
        )
        self._summary_writer.add_histogram("trainer/v-target", v_targets, global_step=self._global_step)
        log_aggregates("game-length", game_lengths)
        log_aggregates("pi-target-entropy", pi_target_entropy)

    def _update_nn(self, dataset: TrainingDataset) -> None:
        def train_epoch() -> tuple[float, float, float]:
            self._nn.train()
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            for x, pi_mask, target_pi, target_value, weight in train_dataloader:
                self._optimizer.zero_grad()
                current_pi_unsoftmaxed, current_value = self._nn(x)
                value_loss = compute_value_loss(current_value, target_value, weight)
                policy_loss = compute_policy_loss(current_pi_unsoftmaxed, pi_mask, target_pi, weight)
                entropy_loss = compute_entropy_loss(current_pi_unsoftmaxed, pi_mask, weight)
                loss = (
                    self._policy_weight * policy_loss
                    + self._value_weight * value_loss
                    + self._entropy_weight * entropy_loss
                )
                loss.backward()
                self._optimizer.step()
                epoch_value_loss += value_loss.item() / len(train_dataloader)
                epoch_policy_loss += policy_loss.item() / len(train_dataloader)
                epoch_entropy_loss += entropy_loss.item() / len(train_dataloader)
            return epoch_value_loss, epoch_policy_loss, epoch_entropy_loss

        def compute_value_loss(
            current_value: torch.Tensor, target_value: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            unweighted_loss = torch.nn.functional.mse_loss(current_value, target_value, reduction="none")
            return (unweighted_loss * weight).sum() / weight.sum()

        def compute_policy_loss(
            pi_tensor_unsoftmaxed: torch.Tensor, pi_mask: torch.Tensor, target_pi: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            policy_loss_unmasked = -target_pi * self._policy_log_softmax(pi_tensor_unsoftmaxed, pi_mask)
            policy_loss_unnormalized = torch.where(pi_mask, policy_loss_unmasked, 0).reshape((pi_mask.shape[0], -1))
            n_actions = pi_mask.reshape((pi_mask.shape[0], -1)).sum(dim=1, keepdim=True)
            policy_loss_unweighted = (policy_loss_unnormalized / n_actions).sum(dim=1)
            return (weight * policy_loss_unweighted).sum() / weight.sum()

        def compute_entropy_loss(
            pi_tensor_unsoftmaxed: torch.Tensor, pi_mask: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            pi = self._policy_softmax(pi_tensor_unsoftmaxed, pi_mask)
            pi_masked = torch.where(pi_mask, pi, 0)
            sample_entropy_unweighted = (
                (pi_masked * torch.log(pi_masked.clip(1e-6))).reshape((pi_mask.shape[0], -1)).sum(dim=1)
            )
            return (weight * sample_entropy_unweighted).sum() / weight.sum()

        @torch.no_grad()
        def epoch_eval() -> tuple[float, float, float]:
            self._nn.eval()
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            for x, pi_mask, target_pi, target_value, weight in test_dataloader:
                current_pi_unsoftmaxed, current_value = self._nn(x)
                value_loss = compute_value_loss(current_value, target_value, weight)
                policy_loss = compute_policy_loss(current_pi_unsoftmaxed, pi_mask, target_pi, weight)
                entropy_loss = compute_entropy_loss(current_pi_unsoftmaxed, pi_mask, weight)
                epoch_value_loss += value_loss.mean().item() / len(test_dataloader)
                epoch_policy_loss += policy_loss.mean().item() / len(test_dataloader)
                epoch_entropy_loss += entropy_loss.mean().item() / len(test_dataloader)
            if self._summary_writer is not None:
                self._summary_writer.add_scalar("eval/policy-loss", epoch_policy_loss, global_step=self._eval_step)
                self._summary_writer.add_scalar("eval/value-loss", epoch_value_loss, global_step=self._eval_step)
                self._summary_writer.add_scalar("eval/entropy-loss", epoch_entropy_loss, global_step=self._eval_step)
                self._eval_step += 1
            return epoch_value_loss, epoch_policy_loss, epoch_entropy_loss

        train_data, test_data = dataset.split(0.9)
        train_dataloader = DataLoader(
            train_data,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=self._batch_size,
            drop_last=False,
        )

        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        with tqdm(desc="nn-update", total=self._epochs_per_update) as pbar:
            for _ in range(self._epochs_per_update):
                train_epoch()
                epoch_test_value_loss, epoch_test_policy_loss, epoch_test_entropy_loss = epoch_eval()
                total_value_loss += epoch_test_value_loss / self._epochs_per_update
                total_policy_loss += epoch_test_policy_loss / self._epochs_per_update
                total_entropy_loss += epoch_test_entropy_loss / self._epochs_per_update
                pbar.set_postfix(
                    {
                        "pi": round(epoch_test_value_loss, 5),
                        "v": round(epoch_test_policy_loss, 5),
                        "e": round(epoch_test_entropy_loss, 5),
                    }
                )
                pbar.update(1)
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("trainer/policy-loss", total_policy_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/value-loss", total_value_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/entropy-loss", total_entropy_loss, global_step=self._global_step)

    @torch.no_grad()
    def _evaluate(self, dataset: TrainingDataset) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
        - the policy logits (mcts format)
        - the policy logits (tensor format)
        - the values
        """
        self._nn.eval()
        dataloader = DataLoader(dataset, batch_size=self._batch_size, drop_last=False)
        pi_logits, vs = [], []
        with tqdm(desc="nn-eval/epoch", total=len(dataset)) as pbar:
            for x, pi_mask_batch, _, _, _ in dataloader:
                pi_tensor_batch, value_batch = self._nn(x)
                for pi_tensor_unsoftmaxed, pi_mask in zip(pi_tensor_batch, pi_mask_batch):
                    pi_vec = pi_tensor_unsoftmaxed[*torch.nonzero(pi_mask.squeeze()).T].ravel()
                    pi_logit = torch.nn.functional.log_softmax(pi_vec, dim=0)
                    pi_logits.append(pi_logit)
                vs.extend(value_batch)
                pbar.update(x.shape[0])
        return torch.concat(pi_logits, dim=0), torch.as_tensor(vs)
