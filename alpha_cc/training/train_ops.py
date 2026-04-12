from collections.abc import Callable

import torch

from alpha_cc.nn.blocks import PolicyLogSoftmax, PolicySoftmax


def compute_wdl_loss(
    wdl_logits: torch.Tensor,
    target_wdl: torch.Tensor,
    weight: torch.Tensor,
    is_internal: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between predicted WDL (from logits) and target WDL distribution."""
    mask = (~is_internal).float()
    log_probs = torch.nn.functional.log_softmax(wdl_logits, dim=-1)
    ce = -(target_wdl * log_probs).sum(dim=-1)
    weight_masked = weight * mask
    return (ce * weight_masked).sum() / weight_masked.sum().clamp_min(1e-8)


def compute_policy_loss(
    pi_tensor_unsoftmaxed: torch.Tensor,
    pi_mask: torch.Tensor,
    target_pi: torch.Tensor,
    weight: torch.Tensor,
    policy_log_softmax: PolicyLogSoftmax,
) -> torch.Tensor:
    batch_size = pi_mask.shape[0]
    pi_mask_flat = pi_mask.reshape(batch_size, -1)
    target_pi_flat = target_pi.reshape(batch_size, -1)
    log_pi = policy_log_softmax(pi_tensor_unsoftmaxed, pi_mask).reshape(batch_size, -1)
    policy_loss_unmasked = -target_pi_flat * log_pi
    policy_loss_unnormalized = torch.where(pi_mask_flat, policy_loss_unmasked, 0)
    n_actions = pi_mask_flat.sum(dim=1, keepdim=True)
    policy_loss_unweighted = (policy_loss_unnormalized / n_actions).sum(dim=1)
    return (weight * policy_loss_unweighted).sum() / weight.sum()


def compute_entropy_loss(
    pi_tensor_unsoftmaxed: torch.Tensor,
    pi_mask: torch.Tensor,
    weight: torch.Tensor,
    policy_softmax: PolicySoftmax,
) -> torch.Tensor:
    batch_size = pi_mask.shape[0]
    pi_mask_flat = pi_mask.reshape(batch_size, -1)
    pi = policy_softmax(pi_tensor_unsoftmaxed, pi_mask).reshape(batch_size, -1)
    pi_masked = torch.where(pi_mask_flat, pi, 0)
    sample_entropy_unweighted = (pi_masked * torch.log(pi_masked.clip(1e-6))).sum(dim=1)
    return (weight * sample_entropy_unweighted).sum() / weight.sum()


def make_train_step(
    nn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_log_softmax: PolicyLogSoftmax,
    policy_softmax: PolicySoftmax,
    policy_weight: float,
    value_weight: float,
    entropy_weight: float,
    mode: str | None = None,
) -> Callable:
    """Create a (optionally compiled) training step that captures the full
    zero_grad → forward → loss → backward → optimizer.step loop as one unit."""

    def train_step(
        x: torch.Tensor,
        pi_mask: torch.Tensor,
        target_pi: torch.Tensor,
        target_wdl: torch.Tensor,
        weight: torch.Tensor,
        is_internal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        pi, wdl_logits = nn(x)
        wdl_loss = compute_wdl_loss(wdl_logits, target_wdl, weight, is_internal)
        policy_loss = compute_policy_loss(pi, pi_mask, target_pi, weight, policy_log_softmax)
        entropy_loss = compute_entropy_loss(pi, pi_mask, weight, policy_softmax)
        loss = policy_weight * policy_loss + value_weight * wdl_loss + entropy_weight * entropy_loss
        loss.backward()
        optimizer.step()
        return wdl_loss, policy_loss, entropy_loss, pi, wdl_logits

    return torch.compile(train_step, mode=mode) if mode is not None else train_step
