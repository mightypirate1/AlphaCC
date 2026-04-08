import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.nn.blocks import PolicyLogSoftmax, PolicySoftmax
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.training.training_dataset import TrainingDataset

logger = logging.getLogger(__name__)


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
        num_dataloader_workers: int = 2,
        lr: float = 1e-4,
        device: str = "cpu",
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._nn = nn.to(device)
        self._compiled_nn = self._nn  # replaced by compile() if called
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight
        self._epochs_per_update = epochs_per_update
        self._batch_size = batch_size
        self._num_dataloader_workers = num_dataloader_workers
        self._device = torch.device(device)
        self._policy_log_softmax = PolicyLogSoftmax(board_size)
        self._policy_softmax = PolicySoftmax(board_size)
        self._optimizer = torch.optim.AdamW(nn.parameters(), lr=lr, weight_decay=l2_reg)
        self._global_step = 0
        self._eval_step = 0
        self._summary_writer = summary_writer

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def compile(self, board_size: int, mode: str = "max-autotune") -> None:
        """JIT-compile the model for faster training. The original model
        is preserved at `self.nn` for ONNX export and state_dict access."""
        logger.info(f"Compiling model with torch.compile(mode={mode!r})...")
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")
        self._compiled_nn = torch.compile(self._nn, mode=mode)  # type: ignore
        # Warmup: run a dummy forward+backward to trigger compilation
        dummy = torch.zeros(self._batch_size, 2, board_size, board_size, device=self._device)
        out_pi, out_wdl = self._compiled_nn(dummy)
        (out_pi.sum() + out_wdl.sum()).backward()
        self._optimizer.zero_grad()
        logger.info("Model compiled and warmed up")

    def train(self, dataset: TrainingDataset) -> tuple[np.ndarray, np.ndarray]:
        kl_divs, td_errors = self._update_nn(dataset)
        self._global_step += 1
        return kl_divs, td_errors

    def set_steps(self, global_step: int, eval_step: int) -> None:
        self._global_step = global_step
        self._eval_step = eval_step

    def get_steps(self) -> tuple[int, int]:
        return self._global_step, self._eval_step

    def set_lr(self, lr: float) -> None:
        for g in self._optimizer.param_groups:
            g["lr"] = lr

    def report_rollout_stats(self, training_datas: list[TrainingData], limit: int | None = None) -> None:
        def log_aggregates(key: str, data: np.ndarray | torch.Tensor, prefix: str = "train-rollouts") -> None:
            if self._summary_writer is None:
                return
            self._summary_writer.add_scalar(f"{prefix}/{key}-mean", data.mean(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"{prefix}/{key}-min", data.min(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"{prefix}/{key}-max", data.max(), global_step=self._global_step)

        def entropy(p: torch.Tensor) -> torch.Tensor:
            return -(p * (p.clamp_min(1e-6).log())).sum(dim=-1)

        def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            return torch.sum(p * (torch.log(p.clip(1e-6)) - torch.log(q.clip(1e-6))), dim=-1)

        if self._summary_writer is None:
            return

        #######
        ### evaluation
        #####
        trajectories = [data.trajectory for data in training_datas]
        experiences = [exp for traj in trajectories for exp in traj]
        if limit is not None:
            experiences = experiences[:limit]
        eval_dataset = TrainingDataset(experiences)
        with torch.no_grad():
            self._nn.eval()
            dataloader = DataLoader(
                eval_dataset,
                batch_size=self._batch_size,
                drop_last=False,
                pin_memory=True,
                shuffle=False,
            )
            pi_logits_list: list[torch.Tensor] = []  # raw logits (legal moves only)
            pi_logprobs: list[torch.Tensor] = []  # log probs over legal moves
            pi_preds: list[torch.Tensor] = []  # probs over legal moves
            pi_targets: list[torch.Tensor] = []  # target probs over legal moves (masked & re-normalized)
            wdl_preds = []
            with tqdm(desc="nn-eval/epoch", total=len(eval_dataset)) as pbar:
                processed = 0
                for batch in dataloader:
                    x, pi_mask_batch, pi_target_batch, _, _, _ = (b.to(self._device) for b in batch)
                    pi_tensor_batch, wdl_logits_batch = self._compiled_nn(x)
                    wdl_preds.append(torch.nn.functional.softmax(wdl_logits_batch, dim=-1).cpu())
                    # iterate per sample
                    for pi_tensor_unsoftmaxed, pi_mask, pi_target_full in zip(
                        pi_tensor_batch, pi_mask_batch, pi_target_batch
                    ):
                        # legal indices
                        legal_idx = torch.nonzero(pi_mask.squeeze(), as_tuple=True)
                        raw_pi_logits = pi_tensor_unsoftmaxed[legal_idx].ravel()
                        target_legal = pi_target_full[legal_idx].ravel()
                        # re-normalize target in case of numerical drift
                        target_legal = target_legal / target_legal.sum().clamp_min(1e-8)

                        pi_logprob = torch.nn.functional.log_softmax(raw_pi_logits, dim=0)
                        pi_pred = torch.nn.functional.softmax(raw_pi_logits, dim=0)

                        pi_logits_list.append(raw_pi_logits.cpu())
                        pi_logprobs.append(pi_logprob.cpu())
                        pi_preds.append(pi_pred.cpu())
                        pi_targets.append(target_legal.cpu())

                    processed += x.shape[0]
                    pbar.update(x.shape[0])

            # flatten (concatenate) for histograms
            pi_logprobs_raveled = torch.cat(pi_logprobs, dim=0)  # pred logprobs (legal)
            pi_targets_raveled = torch.cat(pi_targets, dim=0)  # target probs (legal)
            wdl_pred = torch.cat(wdl_preds, dim=0)  # (N, 3)

        #######
        ### trajectories
        #####
        game_lengths = np.array([len(traj) for traj in trajectories])
        game_ended_early = np.array([traj[-1].game_ended_early for traj in trajectories if traj])
        wdl_targets = np.array([e.wdl_target for e in experiences])
        wdl_target_values = wdl_targets[:, 0] - wdl_targets[:, 2]  # expected value = W - L
        pi_targets_logprobs_raveled = pi_targets_raveled.clamp(1e-6).log()
        pi_target_entropy = torch.as_tensor([entropy(pi_target) for pi_target in pi_targets])
        kl_divergences = torch.as_tensor(
            [kl_divergence(pi_target, pi_pred) for pi_target, pi_pred in zip(pi_targets, pi_preds)]
        )
        self._summary_writer.add_histogram(
            "trainer/pi/pi-target-entropy", pi_target_entropy, global_step=self._global_step
        )
        self._summary_writer.add_histogram(
            "trainer/pi/pi-pred-logprobs", pi_logprobs_raveled, global_step=self._global_step
        )
        self._summary_writer.add_histogram(
            "trainer/pi/pi-target-logprobs", pi_targets_logprobs_raveled, global_step=self._global_step
        )
        wdl_pred_values = wdl_pred[:, 0] - wdl_pred[:, 2]  # expected value = W - L
        self._summary_writer.add_histogram("trainer/value/v-pred", wdl_pred_values, global_step=self._global_step)
        self._summary_writer.add_histogram("trainer/value/v-target", wdl_target_values, global_step=self._global_step)
        self._summary_writer.add_histogram("trainer/wdl/win-pred", wdl_pred[:, 0], global_step=self._global_step)
        self._summary_writer.add_histogram("trainer/wdl/draw-pred", wdl_pred[:, 1], global_step=self._global_step)
        self._summary_writer.add_histogram("trainer/wdl/loss-pred", wdl_pred[:, 2], global_step=self._global_step)
        self._summary_writer.add_scalar(
            "trainer/pi/kl-divergence-mean", kl_divergences.mean(), global_step=self._global_step
        )
        self._summary_writer.add_histogram("trainer/pi/kl-divergence", kl_divergences, global_step=self._global_step)
        log_aggregates("game-length", game_lengths)
        log_aggregates("pi-target-entropy", pi_target_entropy)
        self._summary_writer.add_scalar(
            "train-rollouts/frac-ended-early",
            game_ended_early.mean(),
            global_step=self._global_step,
        )
        winners = np.array([data.winner for data in training_datas])
        n_games = max(len(winners), 1)
        self._summary_writer.add_histogram("train-rollouts/winners", winners, global_step=self._global_step)
        self._summary_writer.add_scalar(
            "train-rollouts/frac-p1-wins", (winners == 1).sum() / n_games, global_step=self._global_step
        )
        self._summary_writer.add_scalar(
            "train-rollouts/frac-p2-wins", (winners == 2).sum() / n_games, global_step=self._global_step
        )
        self._summary_writer.add_scalar(
            "train-rollouts/frac-draws", (winners == 0).sum() / n_games, global_step=self._global_step
        )

        #######
        ### internal nodes
        #####
        internal_nodes = [data.internal_nodes for data in training_datas]
        n_internal_nodes = sum(len(nodes) for nodes in internal_nodes)
        visit_counts = np.array([np.sum(node.n) for nodes in internal_nodes for node in nodes.values()])
        frac_internal_nodes = n_internal_nodes / max(1, len(experiences) + n_internal_nodes)
        if visit_counts.size > 0:
            self._summary_writer.add_histogram(
                "trainer/internal-nodes/visit-counts", visit_counts, global_step=self._global_step
            )
        self._summary_writer.add_scalar(
            "trainer/internal-nodes/frac-internal-nodes", frac_internal_nodes, global_step=self._global_step
        )

        #######
        ### worker fetch stats
        #####
        worker_stats_list = [td.worker_stats for td in training_datas if td.worker_stats.total_fetches > 0]
        if worker_stats_list:
            self._report_worker_stats(worker_stats_list)

    @torch.no_grad()
    def _compute_sample_errors_batched(self, dataset: TrainingDataset) -> tuple[np.ndarray, np.ndarray]:
        """
        Batched forward pass over dataset to compute per-sample KL divergence and TD error.
        Used by PER to update priorities after training.

        TD error is |v_pred - v_target| where v = W - L (expected value from WDL).
        Returns (kl_divs, td_errors) as float32 numpy arrays, one entry per sample.
        Internal nodes get td_error=0 (WDL head is not trained on them).
        """
        self._nn.eval()
        kl_divs_list: list[torch.Tensor] = []
        wdl_errors_list: list[torch.Tensor] = []
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self._num_dataloader_workers,
            prefetch_factor=3,
        )

        for batch in dataloader:
            x, pi_mask_batch, pi_target_batch, target_wdl, weight, is_internal = (b.to(self._device) for b in batch)
            pi_tensor_batch, wdl_logits_batch = self._compiled_nn(x)

            # Per-sample TD error from expected values: |v_pred - v_target| where v = W - L.
            # 0 for internal nodes, dampened by weight for unfinished games.
            wdl_probs = torch.nn.functional.softmax(wdl_logits_batch, dim=-1)
            v_pred = wdl_probs[:, 0] - wdl_probs[:, 2]
            v_target = target_wdl[:, 0] - target_wdl[:, 2]
            td_err = torch.abs(v_pred - v_target)
            td_err = torch.where(is_internal.squeeze(), torch.zeros_like(td_err), td_err)
            td_err = td_err * weight.squeeze()
            wdl_errors_list.append(td_err.cpu())

            # Per-sample KL divergence (batched): KL(pi_target || pi_pred) over legal moves
            log_pred = self._policy_log_softmax(pi_tensor_batch, pi_mask_batch)
            log_target = torch.log(pi_target_batch.clamp_min(1e-6))
            kl_per_sample = pi_target_batch * (log_target - log_pred)
            kl_per_sample = torch.where(pi_mask_batch, kl_per_sample, 0.0)
            kl_per_sample = kl_per_sample.reshape(x.shape[0], -1).sum(dim=1)
            kl_divs_list.append(kl_per_sample.cpu())

        kl_array = torch.cat(kl_divs_list).numpy().astype(np.float32)
        wdl_array = torch.cat(wdl_errors_list).numpy().astype(np.float32)
        return kl_array, wdl_array

    def _report_worker_stats(self, stats_list: list[WorkerStats]) -> None:
        if self._summary_writer is None:
            return
        total_fetches = sum(s.total_fetches for s in stats_list)
        total_fetch_time_us = sum(s.total_fetch_time_us for s in stats_list)

        if total_fetches > 0:
            self._summary_writer.add_scalar(
                "worker/mean-fetch-latency-ms",
                (total_fetch_time_us / total_fetches) / 1000.0,
                global_step=self._global_step,
            )

    def _update_nn(self, dataset: TrainingDataset) -> tuple[np.ndarray, np.ndarray]:
        self._reset_gradient_stats()

        def train_epoch(dataloader: DataLoader) -> tuple[float, float, float]:
            self._nn.train()
            # accumulator of: wdl_loss, policy_loss, entropy_loss
            loss_accumulator = torch.zeros(3, dtype=torch.float64, device="cpu")  # Higher precision for accumulation
            samples_seen = 0
            for data_tuple in dataloader:
                x, pi_mask, target_pi, target_wdl, weight, is_internal = (
                    data.to(self._device) for data in data_tuple
                )
                self._optimizer.zero_grad(set_to_none=True)
                current_pi_unsoftmaxed, current_wdl_logits = self._compiled_nn(x)
                wdl_loss = compute_wdl_loss(current_wdl_logits, target_wdl, weight, is_internal)
                policy_loss = compute_policy_loss(current_pi_unsoftmaxed, pi_mask, target_pi, weight)
                entropy_loss = compute_entropy_loss(current_pi_unsoftmaxed, pi_mask, weight)
                loss = (
                    self._policy_weight * policy_loss
                    + self._value_weight * wdl_loss
                    + self._entropy_weight * entropy_loss
                )
                loss.backward()
                self._accumulate_gradient_stats()
                self._optimizer.step()
                batch_size = x.shape[0]
                samples_seen += batch_size
                batch_losses = torch.tensor(
                    [
                        wdl_loss.detach().cpu().item(),
                        policy_loss.detach().cpu().item(),
                        entropy_loss.detach().cpu().item(),
                    ],
                    dtype=torch.float64,
                )
                loss_accumulator += batch_losses * batch_size
            epoch_wdl_loss, epoch_policy_loss, epoch_entropy_loss = tuple((loss_accumulator / samples_seen).tolist())
            return epoch_wdl_loss, epoch_policy_loss, epoch_entropy_loss

        def compute_wdl_loss(
            wdl_logits: torch.Tensor, target_wdl: torch.Tensor, weight: torch.Tensor, is_internal: torch.Tensor
        ) -> torch.Tensor:
            """Cross-entropy between predicted WDL (from logits) and target WDL distribution."""
            mask = (~is_internal).float()
            log_probs = torch.nn.functional.log_softmax(wdl_logits, dim=-1)
            # Per-sample cross-entropy: -sum(target * log_pred)
            ce = -(target_wdl * log_probs).sum(dim=-1)
            weight_masked = weight * mask
            return (ce * weight_masked).sum() / weight_masked.sum().clamp_min(1e-8)

        def compute_policy_loss(
            pi_tensor_unsoftmaxed: torch.Tensor, pi_mask: torch.Tensor, target_pi: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
            pi_mask_flat = pi_mask.reshape((pi_mask.shape[0], -1))
            policy_loss_unmasked = -target_pi * self._policy_log_softmax(pi_tensor_unsoftmaxed, pi_mask)
            policy_loss_unnormalized = torch.where(pi_mask, policy_loss_unmasked, 0).reshape((pi_mask.shape[0], -1))
            n_actions = pi_mask_flat.sum(dim=1, keepdim=True)
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
            epoch_wdl_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_is_internal = 0.0
            for data_tuple in test_dataloader:
                x, pi_mask, target_pi, target_wdl, weight, is_internal = (
                    data.to(self._device) for data in data_tuple
                )
                current_pi_unsoftmaxed, current_wdl_logits = self._compiled_nn(x)
                wdl_loss = compute_wdl_loss(current_wdl_logits, target_wdl, weight, is_internal)
                policy_loss = compute_policy_loss(current_pi_unsoftmaxed, pi_mask, target_pi, weight)
                entropy_loss = compute_entropy_loss(current_pi_unsoftmaxed, pi_mask, weight)
                epoch_wdl_loss += wdl_loss.mean().cpu().item() / len(test_dataloader)
                epoch_policy_loss += policy_loss.mean().cpu().item() / len(test_dataloader)
                epoch_entropy_loss += entropy_loss.mean().cpu().item() / len(test_dataloader)
                epoch_is_internal += is_internal.float().mean().cpu().item() / len(test_dataloader)
            if self._summary_writer is not None:
                self._summary_writer.add_scalar("eval/policy-loss", epoch_policy_loss, global_step=self._eval_step)
                self._summary_writer.add_scalar("eval/wdl-loss", epoch_wdl_loss, global_step=self._eval_step)
                self._summary_writer.add_scalar("eval/entropy-loss", epoch_entropy_loss, global_step=self._eval_step)
                self._summary_writer.add_scalar(
                    "eval/effective-frac-internal", epoch_is_internal, global_step=self._eval_step
                )
                self._eval_step += 1
            return epoch_wdl_loss, epoch_policy_loss, epoch_entropy_loss

        train_data, test_data = dataset.split(0.9)
        test_dataloader = DataLoader(
            test_data,
            batch_size=self._batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self._num_dataloader_workers,
            prefetch_factor=3,
        )

        total_wdl_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        with tqdm(desc="nn-update", total=self._epochs_per_update) as pbar:
            for _ in range(self._epochs_per_update):
                train_dataloader = DataLoader(
                    train_data,
                    batch_size=self._batch_size,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=True,
                    num_workers=self._num_dataloader_workers,
                    prefetch_factor=3,
                )
                train_epoch(train_dataloader)
                epoch_test_wdl_loss, epoch_test_policy_loss, epoch_test_entropy_loss = epoch_eval()
                total_wdl_loss += epoch_test_wdl_loss / self._epochs_per_update
                total_policy_loss += epoch_test_policy_loss / self._epochs_per_update
                total_entropy_loss += epoch_test_entropy_loss / self._epochs_per_update
                pbar.set_postfix(
                    {
                        "pi": round(epoch_test_policy_loss, 5),
                        "wdl": round(epoch_test_wdl_loss, 5),
                        "e": round(epoch_test_entropy_loss, 5),
                    }
                )
                pbar.update(1)
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("trainer/policy-loss", total_policy_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/wdl-loss", total_wdl_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/entropy-loss", total_entropy_loss, global_step=self._global_step)

        self._flush_gradient_stats()

        # Final batched eval pass over full dataset for PER priority updates
        kl_divs, td_errors = self._compute_sample_errors_batched(dataset)
        return kl_divs, td_errors

    _PARAM_GROUPS = ("encoder", "global-encoder", "local-encoder", "policy-head", "value-head")

    def _get_param_group_modules(self) -> dict[str, torch.nn.Module]:
        return {
            "encoder": self._nn._encoder,
            "global-encoder": self._nn._global_encoder,
            "local-encoder": self._nn._local_encoder,
            "policy-head": self._nn._policy_head,
            "value-head": self._nn._value_combined,
        }

    def _reset_gradient_stats(self) -> None:
        self._grad_stats: dict[str, dict[str, float]] = {
            name: {
                "grad-norm-sq": 0.0,
                "grad-max": 0.0,
                "grad-min": float("inf"),
                "weight-norm-sq": 0.0,
                "weight-max": 0.0,
            }
            for name in self._PARAM_GROUPS
        }

    @torch.no_grad()
    def _accumulate_gradient_stats(self) -> None:
        if not hasattr(self, "_grad_stats"):
            self._reset_gradient_stats()
        for name, module in self._get_param_group_modules().items():
            stats = self._grad_stats[name]
            w_norm_sq = 0.0
            g_norm_sq = 0.0
            for p in module.parameters():
                w_norm_sq += p.data.norm().item() ** 2
                stats["weight-max"] = max(stats["weight-max"], p.data.abs().max().item())
                if p.grad is not None:
                    g_norm_sq += p.grad.norm().item() ** 2
                    stats["grad-max"] = max(stats["grad-max"], p.grad.abs().max().item())
                    stats["grad-min"] = min(stats["grad-min"], p.grad.abs().min().item())
            stats["grad-norm-sq"] = max(stats["grad-norm-sq"], g_norm_sq)
            stats["weight-norm-sq"] = max(stats["weight-norm-sq"], w_norm_sq)

    def _flush_gradient_stats(self) -> None:
        if self._summary_writer is None or not hasattr(self, "_grad_stats"):
            return
        step = self._global_step
        for name, stats in self._grad_stats.items():
            for key, val in stats.items():
                if key == "grad-min" and val == float("inf"):
                    continue
                if key.endswith("-norm-sq"):
                    val = val**0.5
                    key = key.removesuffix("-sq")
                self._summary_writer.add_scalar(f"gradients/{name}/{key}", val, global_step=step)
        self._reset_gradient_stats()
