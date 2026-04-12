from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from alpha_cc.engine import GameConfig

from alpha_cc.config import Environment
from alpha_cc.training.training_checkpoint import TrainingCheckpoint

logger = logging.getLogger(__name__)


def serialize_model(
    model: torch.nn.Module,
    config: GameConfig,
    compiled_batch_size: int | None = None,
) -> bytes:
    model.eval()
    device = next(model.parameters()).device
    batch = compiled_batch_size or 1
    dummy = torch.zeros(batch, config.state_channels, config.board_size, config.board_size, device=device)
    tmp_path = Path(Environment.model_dir) / "temp.onnx"
    dynamic_axes = (
        None
        if compiled_batch_size is not None
        else {
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
    )
    torch.onnx.export(
        model,
        (dummy,),
        tmp_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    with open(tmp_path, "rb") as f:
        payload = f.read()
    model.train()
    return payload


def load_weights(run_id: str, index: int) -> dict[str, Any]:
    path = save_path(run_id, index)
    if not Path(path).exists():
        raise FileNotFoundError(f"weights file {path} does not exist")
    return torch.load(path, weights_only=True)


def save_weights(run_id: str, curr_index: int, weights: dict[str, Any], onnx_payload: bytes) -> None:
    path = save_path(run_id, curr_index)
    latest_path = save_path_latest(run_id)
    onnx_path = save_path(run_id, curr_index, ext="onnx")
    onnx_latest_path = save_path_latest(run_id, ext="onnx")
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(weights, latest_path)
    torch.save(weights, path)
    Path(onnx_latest_path).write_bytes(onnx_payload)
    Path(onnx_path).write_bytes(onnx_payload)


def load_saved_checkpoint(run_id: str) -> TrainingCheckpoint | None:
    checkpoint_path = save_latest_checkpoint_path(run_id)
    if not Path(checkpoint_path).exists():
        return None
    return TrainingCheckpoint.from_path(checkpoint_path, verbose=True)


def save_path_latest(run_id: str, ext: str = "pth") -> str:
    return f"{save_root(run_id)}/weights-latest.{ext}"


def save_path(run_id: str, index: int, ext: str = "pth") -> str:
    return f"{save_root(run_id)}/weights-{str(index).zfill(4)}.{ext}"


def save_root(run_id: str) -> str:
    return f"{Environment.model_dir}/{run_id}"


def save_latest_checkpoint_path(run_id: str) -> str:
    return f"{save_root(run_id)}/checkpoint-latest.pth"


def save_checkpoint_path(run_id: str, index: int) -> str:
    return f"{save_root(run_id)}/checkpoint-{str(index).zfill(4)}.pth"
