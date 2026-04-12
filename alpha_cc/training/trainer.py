from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.nn.blocks import PolicyLogSoftmax, PolicySoftmax
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.training import train_ops
from alpha_cc.training.training_dataset import TrainingDataset

if TYPE_CHECKING:
    from alpha_cc.engine import GameConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: GameConfig,
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
        self._config = config
        self._policy_log_softmax = PolicyLogSoftmax(config)
        self._policy_softmax = PolicySoftmax(config)
        self._optimizer = torch.optim.AdamW(nn.parameters(), lr=lr, weight_decay=l2_reg)
        self._global_step = 0
        self._eval_step = 0
        self._summary_writer = summary_writer
        self._train_step = train_ops.make_train_step(
            self._nn,
            self._optimizer,
            self._policy_log_softmax,
            self._policy_softmax,
            self._policy_weight,
            self._value_weight,
            self._entropy_weight,
        )

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def compile(self, mode: str = "max-autotune") -> None:
        """JIT-compile the training step and model for faster training. The original
        model is preserved at `self.nn` for ONNX export and state_dict access."""
        logger.info(f"Compiling training step with torch.compile(mode={mode!r})...")
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")
        self._compiled_nn = torch.compile(self._nn, mode=mode)  # type: ignore  # used for eval
        self._train_step = train_ops.make_train_step(
            self._nn,
            self._optimizer,
            self._policy_log_softmax,
            self._policy_softmax,
            self._policy_weight,
            self._value_weight,
            self._entropy_weight,
            mode=mode,
        )
        # Warm up the full training step (forward + backward + optimizer) so compilation
        # completes before workers start, not while they compete for CPU.
        cfg = self._config
        bs, n = cfg.board_size, self._batch_size
        dummy_x = torch.zeros(n, cfg.state_channels, bs, bs, device=self._device)
        dummy_mask = torch.ones(n, *cfg.policy_shape, dtype=torch.bool, device=self._device)
        dummy_pi = torch.full((n, *cfg.policy_shape), 1.0 / cfg.policy_size, device=self._device)
        dummy_wdl = torch.full((n, 3), 1.0 / 3.0, device=self._device)
        dummy_weight = torch.ones(n, device=self._device)
        dummy_is_internal = torch.zeros(n, dtype=torch.bool, device=self._device)
        self._train_step(dummy_x, dummy_mask, dummy_pi, dummy_wdl, dummy_weight, dummy_is_internal)
        logger.info("Training step compiled and warmed up")

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
                batch_size=1024,
                drop_last=False,
                shuffle=False,
                pin_memory=self._device.type == "cuda",
                num_workers=self._num_dataloader_workers,
                prefetch_factor=3 if self._num_dataloader_workers > 0 else None,
            )
            pi_logprobs_list: list[torch.Tensor] = []
            pi_targets_list: list[torch.Tensor] = []
            wdl_preds: list[torch.Tensor] = []
            kl_divs_list: list[torch.Tensor] = []
            entropy_list: list[torch.Tensor] = []
            with tqdm(desc="nn-eval/epoch", total=len(eval_dataset)) as pbar:
                for batch in dataloader:
                    x, pi_mask_batch, pi_target_batch, _, _, _ = (b.to(self._device) for b in batch)
                    batch_size = x.shape[0]
                    pi_tensor_batch, wdl_logits_batch = self._compiled_nn(x)
                    wdl_preds.append(torch.nn.functional.softmax(wdl_logits_batch, dim=-1).cpu())

                    # Flatten spatial dims → (B, N) for fully vectorized ops
                    mask = pi_mask_batch.reshape(batch_size, -1).bool()
                    target = pi_target_batch.reshape(batch_size, -1) * mask
                    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)

                    # Masked log_softmax; replace -inf at illegal positions with 0 to avoid 0*-inf=nan
                    log_pred = torch.nn.functional.log_softmax(
                        pi_tensor_batch.reshape(batch_size, -1).masked_fill(~mask, float("-inf")), dim=1
                    ).masked_fill(~mask, 0.0)
                    log_target = target.clamp_min(1e-6).log()

                    # Legal-only values for histograms (variable-length per sample, concatenated)
                    pi_logprobs_list.append(log_pred[mask].cpu())
                    pi_targets_list.append(target[mask].cpu())
                    # Per-sample entropy and KL
                    entropy_list.append(-(target * log_target).sum(dim=1).cpu())
                    kl_divs_list.append((target * (log_target - log_pred)).sum(dim=1).cpu())

                    pbar.update(batch_size)

            pi_logprobs_raveled = torch.cat(pi_logprobs_list)
            pi_targets_raveled = torch.cat(pi_targets_list)
            wdl_pred = torch.cat(wdl_preds)
            pi_target_entropy = torch.cat(entropy_list)
            kl_divergences = torch.cat(kl_divs_list)

        #######
        ### trajectories
        #####
        game_lengths = np.array([len(traj) for traj in trajectories])
        game_ended_early = np.array([traj[-1].game_ended_early for traj in trajectories if traj])
        wdl_targets = np.array([e.wdl_target for e in experiences])
        wdl_target_values = wdl_targets[:, 0] - wdl_targets[:, 2]  # expected value = W - L
        pi_targets_logprobs_raveled = pi_targets_raveled.clamp(1e-6).log()
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
        wdl_entropy = -(wdl_pred * wdl_pred.clamp_min(1e-6).log()).sum(dim=1)
        self._summary_writer.add_histogram("trainer/wdl/entropy", wdl_entropy, global_step=self._global_step)
        log_aggregates("wdl-entropy", wdl_entropy)
        self._summary_writer.add_scalar(
            "trainer/pi/kl-divergence-mean", kl_divergences.mean(), global_step=self._global_step
        )
        self._summary_writer.add_histogram("trainer/pi/kl-divergence", kl_divergences, global_step=self._global_step)
        log_aggregates("game-length", game_lengths)
        terminal_wdl = np.array([traj[-1].wdl_target for traj in trajectories if traj])
        terminal_abs_value = np.abs(terminal_wdl[:, 0] - terminal_wdl[:, 2])
        log_aggregates("terminal-abs-value", terminal_abs_value)
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

        def train_epoch(
            dataloader: DataLoader, collect: bool = False
        ) -> tuple[float, float, float, list[torch.Tensor], list[torch.Tensor]]:
            self._nn.train()
            # accumulator of: wdl_loss, policy_loss, entropy_loss
            loss_accumulator = torch.zeros(3, dtype=torch.float64, device="cpu")  # Higher precision for accumulation
            samples_seen = 0
            kl_list: list[torch.Tensor] = []
            td_list: list[torch.Tensor] = []
            with tqdm(desc="nn-update", total=len(dataset)) as pbar:
                for data_tuple in dataloader:
                    x, pi_mask, target_pi, target_wdl, weight, is_internal = (
                        data.to(self._device) for data in data_tuple
                    )
                    # Pad tail batch to batch_size so compiled step sees fixed shapes.
                    n_real = x.shape[0]
                    if n_real < self._batch_size:
                        n_pad = self._batch_size - n_real

                        def _pad(t: torch.Tensor, padding: int, value: float = 0.0) -> torch.Tensor:
                            return F.pad(t.float(), (0, 0) * (t.dim() - 1) + (0, padding), value=value).to(t.dtype)

                        x = _pad(x, n_pad)
                        pi_mask = _pad(pi_mask, n_pad, value=1.0)
                        target_pi = _pad(target_pi, n_pad)
                        target_wdl = _pad(target_wdl, n_pad, value=1.0 / 3.0)
                        weight = _pad(weight, n_pad)
                        is_internal = _pad(is_internal, n_pad)
                    wdl_loss, policy_loss, entropy_loss, current_pi_unsoftmaxed, current_wdl_logits = self._train_step(
                        x, pi_mask, target_pi, target_wdl, weight, is_internal
                    )
                    self._accumulate_gradient_stats()
                    if collect:
                        with torch.no_grad():
                            # TD error: |v_pred - v_target|, 0 for internal, weighted by sample weight
                            wdl_probs = torch.nn.functional.softmax(current_wdl_logits[:n_real], dim=-1)
                            v_pred = wdl_probs[:, 0] - wdl_probs[:, 2]
                            v_target = target_wdl[:n_real, 0] - target_wdl[:n_real, 2]
                            td_err = torch.abs(v_pred - v_target)
                            td_err = torch.where(is_internal[:n_real].squeeze(), torch.zeros_like(td_err), td_err)
                            td_list.append((td_err * weight[:n_real].squeeze()).cpu())
                            # KL(target || pred) over legal moves, unweighted
                            log_pred = self._policy_log_softmax(
                                current_pi_unsoftmaxed[:n_real], pi_mask[:n_real]
                            ).reshape(n_real, -1)
                            target_flat = target_pi[:n_real].reshape(n_real, -1)
                            mask_flat = pi_mask[:n_real].reshape(n_real, -1)
                            log_target = torch.log(target_flat.clamp_min(1e-6))
                            kl = target_flat * (log_target - log_pred)
                            kl_list.append(torch.where(mask_flat, kl, 0.0).sum(dim=1).cpu())
                    batch_size = n_real
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
                    pbar.set_postfix(
                        {
                            "pi": round(policy_loss.detach().cpu().item(), 5),
                            "wdl": round(wdl_loss.detach().cpu().item(), 5),
                            "e": round(entropy_loss.detach().cpu().item(), 5),
                        }
                    )
                    pbar.update(batch_size)
                epoch_wdl_loss, epoch_policy_loss, epoch_entropy_loss = tuple(
                    (loss_accumulator / samples_seen).tolist()
                )
                return epoch_wdl_loss, epoch_policy_loss, epoch_entropy_loss, kl_list, td_list

        @torch.no_grad()
        def epoch_eval() -> tuple[float, float, float]:
            self._nn.eval()
            epoch_wdl_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_is_internal = 0.0
            for data_tuple in test_dataloader:
                x, pi_mask, target_pi, target_wdl, weight, is_internal = (data.to(self._device) for data in data_tuple)
                current_pi_unsoftmaxed, current_wdl_logits = self._compiled_nn(x)
                wdl_loss = train_ops.compute_wdl_loss(current_wdl_logits, target_wdl, weight, is_internal)
                policy_loss = train_ops.compute_policy_loss(
                    current_pi_unsoftmaxed, pi_mask, target_pi, weight, self._policy_log_softmax
                )
                entropy_loss = train_ops.compute_entropy_loss(
                    current_pi_unsoftmaxed, pi_mask, weight, self._policy_softmax
                )
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

        _, test_data = dataset.split(0.9)
        test_dataloader = DataLoader(
            test_data,
            batch_size=self._batch_size,
            drop_last=True,
            pin_memory=self._device.type == "cuda",
            num_workers=self._num_dataloader_workers,
            prefetch_factor=3 if self._num_dataloader_workers > 0 else None,
        )

        total_wdl_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        kl_divs: np.ndarray | None = None
        td_errors: np.ndarray | None = None
        for epoch_idx in range(self._epochs_per_update):
            is_last = epoch_idx == self._epochs_per_update - 1
            train_dataloader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=not is_last,
                drop_last=not is_last,
                pin_memory=self._device.type == "cuda",
                num_workers=self._num_dataloader_workers,
                prefetch_factor=3 if self._num_dataloader_workers > 0 else None,
            )
            _, _, _, kl_list, td_list = train_epoch(train_dataloader, collect=is_last)
            if is_last:
                kl_divs = torch.cat(kl_list).numpy().astype(np.float32)
                td_errors = torch.cat(td_list).numpy().astype(np.float32)
            epoch_test_wdl_loss, epoch_test_policy_loss, epoch_test_entropy_loss = epoch_eval()
            total_wdl_loss += epoch_test_wdl_loss / self._epochs_per_update
            total_policy_loss += epoch_test_policy_loss / self._epochs_per_update
            total_entropy_loss += epoch_test_entropy_loss / self._epochs_per_update
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("trainer/policy-loss", total_policy_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/wdl-loss", total_wdl_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/entropy-loss", total_entropy_loss, global_step=self._global_step)

        self._flush_gradient_stats()
        return kl_divs, td_errors  # type: ignore # TODO: solve this cleaner

    def _get_param_group_modules(self) -> dict[str, torch.nn.Module]:
        return self._nn.param_groups()

    def _reset_gradient_stats(self) -> None:
        self._grad_stats: dict[str, dict[str, float]] = {
            name: {
                "grad-norm-sq": 0.0,
                "grad-max": 0.0,
                "grad-min": float("inf"),
                "weight-norm-sq": 0.0,
                "weight-max": 0.0,
            }
            for name in self._get_param_group_modules()
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
