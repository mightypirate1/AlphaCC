from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Self

import torch

from alpha_cc.training.training_dataset import TrainingDataset

logger = getLogger(__name__)


@dataclass
class TrainingCheckpoint:
    run_id: str
    model_state_dict: dict[str, torch.Tensor]
    champion_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    current_index: int
    champion_index: int
    replay_buffer: TrainingDataset

    def __repr__(self) -> str:
        return (
            r"TrainingCheckpoint {""\n"
            f"    run_id: {self.run_id},\n"
            f"    current_index: {self.current_index},\n"
            f"    champion_index: {self.champion_index},\n"
            f"    replay_buffer_size: {len(self.replay_buffer)},\n"
            r"}"
        )

    @classmethod
    def from_path(cls: type[Self], path: str, verbose: bool = False) -> Self:
        checkpoint: Self = torch.load(path, weights_only=False)
        if verbose:
            logger.info(f"Loaded checkpoint from {path}:\n{checkpoint}")
        return checkpoint

    def save(self, path: str, verbose: bool = False) -> None:
        if verbose:
            logger.info(f"Saved checkpoint to {path}:\n{self}")
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(self, path)
