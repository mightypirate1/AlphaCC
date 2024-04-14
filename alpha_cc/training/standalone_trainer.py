import torch
from torch.utils.data import DataLoader
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.training.training_dataset import TrainingDataset


class StandaloneTrainer:
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        epochs_per_update: int = 3,
    ) -> None:
        self._agent = agent
        self._board = board
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._epochs_per_update = epochs_per_update
        self._optimizer = torch.optim.Adam(agent.nn.parameters(), lr=1e-4)

    def train(self, num_samples: int = 1000) -> None:
        self._agent.nn.eval()
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            while num_samples > 0:
                trajectory = self.rollout_trajectory()
                trajectories.append(trajectory)
                n = len(trajectory)
                num_samples -= n
                pbar.update(n)
        self._update_nn(trajectories)

    @torch.no_grad()
    def rollout_trajectory(self) -> list[MCTSExperience]:
        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        while not board.board_info.game_over and board.board_info.duration < agent.max_game_length:
            move = agent.choose_move(board, training=True)
            board = board.perform_move(move)

        trajectory = agent.trajectory
        self._agent.on_game_end()
        return trajectory

    def _update_nn(self, trajectories: list[list[MCTSExperience]]) -> None:
        def train_epoch(epoch: int) -> float:
            epoch_loss = 0.0
            for x, target_value, target_pi, pi_mask in tqdm(dataloader, desc=f"nn update: epoch {epoch}"):
                self._optimizer.zero_grad()
                current_pi, current_value = self._agent.nn(x, pi_mask)
                value_loss = torch.nn.functional.mse_loss(current_value, target_value)
                policy_loss = compute_policy_loss(current_pi, target_pi, pi_mask)
                loss = (self._policy_weight * policy_loss + self._value_weight * value_loss).mean()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            return epoch_loss / len(dataloader)

        def compute_policy_loss(
            current_pi: torch.Tensor, target_pi: torch.Tensor, pi_mask: torch.Tensor
        ) -> torch.Tensor:
            policy_loss_unmasked = -target_pi * torch.log(current_pi + 1e-6)
            return torch.where(pi_mask, policy_loss_unmasked, 0.0)

        self._agent.nn.train()
        self._agent.nn.clear_cache()
        dataset = TrainingDataset(trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )

        total_loss = 0.0
        for epoch in range(1, self._epochs_per_update + 1):
            epoch_loss = train_epoch(epoch)
            total_loss += epoch_loss / self._epochs_per_update
            print(epoch_loss)  # noqa
        print(total_loss)  # noqa
