from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch

from alpha_cc.training import TrainingDataset


@dataclass
class TrainingCheckpoint:
    run_id: str
    model_state_dict: dict[str, torch.Tensor]
    champion_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    current_index: int
    champion_index: int
    replay_buffer: TrainingDataset

    @classmethod
    def from_path(cls: type[Self], path: str) -> Self:
        checkpoint: Self = torch.load(path)
        return checkpoint

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(self, path)
