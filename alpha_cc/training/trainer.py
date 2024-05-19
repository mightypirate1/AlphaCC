import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.nn.blocks import PolicyLogSoftmax, PolicySoftmax
from alpha_cc.nn.nets import DualHeadNet
from alpha_cc.training.training_dataset import TrainingDataset


class Trainer:
    def __init__(
        self,
        board_size: int,
        nn: DualHeadNet,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.0,
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
        self._optimizer = torch.optim.AdamW(nn.parameters(), lr=lr, weight_decay=1e-4)
        self._global_step = 0
        self._summary_writer = summary_writer

    @property
    def nn(self) -> DualHeadNet:
        return self._nn

    def train(self, dataset: TrainingDataset) -> None:
        self._update_nn(dataset)
        self._global_step += 1

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

        if self._summary_writer is not None:
            game_lengths = np.array([len(traj) for traj in trajectories])
            num_samples = game_lengths.sum()
            v_targets = np.array([e.v_target for traj in trajectories for e in traj])
            pi_targets_flat = np.concatenate([e.pi_target.ravel() for traj in trajectories for e in traj])
            eval_dataset = TrainingDataset([exp for traj in trajectories for exp in traj])
            pi_concatenated, pi_tensor, v = self._evaluate(eval_dataset)
            pi_targets = torch.stack([pi_target for _, _, pi_target, _ in eval_dataset])  # type: ignore
            pi_entropy = entropy(pi_tensor)
            pi_target_entropy = entropy(pi_targets)
            self._summary_writer.add_scalar("trainer/num-samples", num_samples, global_step=self._global_step)
            self._summary_writer.add_histogram("trainer/pi-pred-entropy", pi_entropy, global_step=self._global_step)
            self._summary_writer.add_histogram(
                "trainer/pi-target-entropy", pi_target_entropy, global_step=self._global_step
            )
            self._summary_writer.add_histogram("trainer/pi-pred", pi_concatenated, global_step=self._global_step)
            self._summary_writer.add_histogram("trainer/v-pred", v, global_step=self._global_step)
            self._summary_writer.add_histogram("trainer/pi-target", pi_targets_flat, global_step=self._global_step)
            self._summary_writer.add_histogram("trainer/v-target", v_targets, global_step=self._global_step)
            log_aggregates("game-length", game_lengths)
            log_aggregates("pi-pred-entropy", pi_entropy)
            log_aggregates("pi-target-entropy", pi_target_entropy)

    def _update_nn(self, dataset: TrainingDataset) -> None:
        def train_epoch() -> tuple[float, float, float]:
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            for x, pi_mask, target_pi, target_value in dataloader:
                self._optimizer.zero_grad()
                current_pi_unsoftmaxed, current_value = self._nn(x)
                value_loss = compute_value_loss(current_value, target_value)
                policy_loss = compute_policy_loss(current_pi_unsoftmaxed, pi_mask, target_pi)
                entropy_loss = compute_entropy_loss(current_pi_unsoftmaxed, pi_mask)
                loss = (
                    self._policy_weight * policy_loss
                    + self._value_weight * value_loss
                    + self._entropy_weight * entropy_loss
                )
                loss.backward()
                self._optimizer.step()
                epoch_value_loss += value_loss.item() / len(dataloader)
                epoch_policy_loss += policy_loss.item() / len(dataloader)
                epoch_entropy_loss += entropy_loss.item() / len(dataloader)
            return epoch_value_loss, epoch_policy_loss, epoch_entropy_loss

        def compute_value_loss(current_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(current_value, target_value).mean()

        def compute_policy_loss(
            pi_tensor_unsoftmaxed: torch.Tensor, pi_mask: torch.Tensor, target_pi: torch.Tensor
        ) -> torch.Tensor:
            policy_loss_unmasked = -target_pi * self._policy_log_softmax(pi_tensor_unsoftmaxed, pi_mask)
            policy_loss_unweighted = torch.where(pi_mask, policy_loss_unmasked, 0).reshape((pi_mask.shape[0], -1))
            n_actions = pi_mask.reshape((pi_mask.shape[0], -1)).sum(dim=1, keepdim=True)
            return (policy_loss_unweighted / n_actions).mean()

        def compute_entropy_loss(pi_tensor_unsoftmaxed: torch.Tensor, pi_mask: torch.Tensor) -> torch.Tensor:
            pi = self._policy_softmax(pi_tensor_unsoftmaxed, pi_mask)
            pi_masked = torch.where(pi_mask, pi, 0)
            sample_entropy = (pi_masked * torch.log(pi_masked.clip(1e-6))).reshape((pi_mask.shape[0], -1)).sum(dim=1)
            return sample_entropy.mean()

        self._nn.train()
        self._nn.clear_cache()
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        with tqdm(desc="nn-update", total=self._epochs_per_update) as pbar:
            for _ in range(self._epochs_per_update):
                epoch_value_loss, epoch_policy_loss, epoch_entropy_loss = train_epoch()
                total_value_loss += epoch_value_loss / self._epochs_per_update
                total_policy_loss += epoch_policy_loss / self._epochs_per_update
                total_entropy_loss += epoch_entropy_loss / self._epochs_per_update
                pbar.set_postfix(
                    {
                        "pi": round(epoch_policy_loss, 5),
                        "v": round(epoch_value_loss, 5),
                        "e": round(epoch_value_loss, 5),
                    }
                )
                pbar.update(1)
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("trainer/policy-loss", total_policy_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/value-loss", total_value_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/entropy-loss", total_entropy_loss, global_step=self._global_step)

    @torch.no_grad()
    def _evaluate(self, dataset: TrainingDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._nn.eval()
        dataloader = DataLoader(dataset, batch_size=self._batch_size, drop_last=False)
        pis, pi_tensors, vs = [], [], []
        with tqdm(desc="nn-eval/epoch", total=len(dataset)) as pbar:
            for x, pi_mask_batch, _, _ in dataloader:
                pi_tensor_batch, value_batch = self._nn(x)
                for pi_tensor_unsoftmaxed, pi_mask in zip(pi_tensor_batch, pi_mask_batch):
                    pi_vec = pi_tensor_unsoftmaxed[*torch.nonzero(pi_mask.squeeze()).T].ravel()
                    pi = torch.nn.functional.softmax(pi_vec, dim=0)
                    pis.append(pi)
                pi_tensors.append(self._policy_softmax(pi_tensor_batch, pi_mask_batch))
                vs.extend(value_batch)
                pbar.update(x.shape[0])

        pi = torch.concat(pis, dim=0)
        pi_tensor = torch.concat(pi_tensors, dim=0)
        value = torch.as_tensor(vs)
        return pi, pi_tensor, value
