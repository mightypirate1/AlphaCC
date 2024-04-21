import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.reward import HeuristicReward
from alpha_cc.training.training_dataset import TrainingDataset


class StandaloneTrainer:
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        epochs_per_update: int = 3,
        lr: float = 1e-4,
        apply_heuristic_on_max_game_length: bool = False,
    ) -> None:
        self._agent = agent
        self._board = board
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._epochs_per_update = epochs_per_update
        self._optimizer = torch.optim.Adam(agent.nn.parameters(), lr=lr)
        self._apply_heuristic_on_max_game_length = apply_heuristic_on_max_game_length
        self._heuristic = HeuristicReward(board.size, scale=0.01)

    def train(self, num_samples: int = 1000, max_game_length: int | None = None) -> None:
        self._agent.nn.eval()
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            while num_samples > 0:
                trajectory = self.rollout_trajectory(max_game_length)
                trajectories.append(trajectory)
                n = len(trajectory)
                num_samples -= n
                pbar.update(n)
        self._update_nn(trajectories)

    @torch.no_grad()
    def rollout_trajectory(self, max_game_length: int | None = None) -> list[MCTSExperience]:
        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        def game_exceeds_duration() -> bool:
            if max_game_length is None:
                return False
            return board.board_info.duration > max_game_length

        while not board.board_info.game_over and not game_exceeds_duration():
            move = agent.choose_move(board, training=True)
            board = board.perform_move(move)

        trajectory = self._assign_value_targets(agent.trajectory)
        self._agent.on_game_end()
        return trajectory

    def _assign_value_targets(self, trajectory: list[MCTSExperience]) -> list[MCTSExperience]:
        final_exp = trajectory[-1]
        if self._apply_heuristic_on_max_game_length and np.abs(final_exp.v_target) != 1.0:
            # we don't have deepmind's resources, so we cut games short and evaluate on heuristics
            v_s = self._heuristic(final_exp.state)
            next_state = final_exp.state.children[0]  # arbitrary next state (this is a heuristic after all)
            v_sp = self._heuristic(next_state)
            v = float(v_s - v_sp)
        else:
            v = final_exp.v_target

        for experience in reversed(trajectory):
            experience.v_target = v
            v *= -1.0
        return trajectory

    def _update_nn(self, trajectories: list[list[MCTSExperience]]) -> None:
        def train_epoch(epoch: int) -> tuple[float, float]:
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            with tqdm(total=len(dataset), desc=f"nn-update/epoch {epoch}") as pbar:
                for x, target_value, target_pi, pi_mask in dataloader:
                    self._optimizer.zero_grad()
                    current_pi, current_value = self._agent.nn(x, pi_mask)
                    value_loss = compute_value_loss(current_value, target_value)
                    policy_loss = compute_policy_loss(current_pi, target_pi, pi_mask)
                    loss = policy_loss + value_loss
                    loss.backward()
                    self._optimizer.step()
                    epoch_value_loss += value_loss.item() / len(dataloader)
                    epoch_policy_loss += policy_loss.item() / len(dataloader)
                    pbar.update(x.shape[0])
            return epoch_value_loss, epoch_policy_loss

        def compute_value_loss(current_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
            return self._value_weight * torch.nn.functional.mse_loss(current_value, target_value).mean()

        def compute_policy_loss(
            current_pi: torch.Tensor, target_pi: torch.Tensor, pi_mask: torch.Tensor
        ) -> torch.Tensor:
            policy_loss_unmasked = -target_pi * torch.log(current_pi + 1e-6)
            policy_loss = torch.where(pi_mask, policy_loss_unmasked, 0.0)
            return self._policy_weight * policy_loss.sum() / pi_mask.sum()

        self._agent.nn.train()
        self._agent.nn.clear_cache()
        dataset = TrainingDataset(trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
        )

        total_value_loss = 0.0
        total_policy_loss = 0.0
        for epoch in range(1, self._epochs_per_update + 1):
            epoch_value_loss, epoch_policy_loss = train_epoch(epoch)
            total_value_loss += epoch_value_loss / self._epochs_per_update
            total_policy_loss += epoch_policy_loss / self._epochs_per_update
            print(f"epoch losses: value={epoch_value_loss} policy={epoch_policy_loss}")  # noqa
        print(f"total losses: value={total_value_loss} policy={total_policy_loss}")  # noqa
