import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.engine import Board


class TrainingDataset(Dataset):
    def __init__(self, trajectories: list[list[MCTSExperience]]) -> None:
        self._experiences = [exp for traj in trajectories for exp in traj]

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        exp = self._experiences[index]
        x = torch.as_tensor(exp.state.matrix).unsqueeze(0)
        value_target = torch.as_tensor(exp.v_target)
        pi_mask = torch.as_tensor(exp.state.action_mask).unsqueeze(0)
        pi_target = self._create_pi_target_tensor(exp)
        return x.float(), value_target.float(), pi_target.float(), pi_mask.bool()

    def _create_pi_target_tensor(self, exp: MCTSExperience) -> torch.Tensor:
        """
        The neural net and the MCTS speak different lanugages:
        - nn thinks a policy is a big tensor encoding all moves (including legal ones)
        - mcts things a policy is a vector for only the legal moves

        This is because neural nets need a fixes size output with semantic consistency,
        and MCTS is more natural to write with illegal moves not even considered.

        This method translates mcts-style pi targets into nn pi targets.

        """
        # TODO: make pretty
        pi_target = np.zeros(4 * (exp.state.board.size,))
        for i in range(len(exp.state.children)):
            from_coord, to_coord = exp.state.action_mask_indices[i]
            pi_target[from_coord.x, from_coord.y, to_coord.x, to_coord.y] = exp.pi_target[i]
        return torch.as_tensor(pi_target).unsqueeze(0)


class StandaloneTrainer:
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        gamma: float = 1.0,
    ) -> None:
        self._agent = agent
        self._board = board
        self._gamma = gamma
        self._optimizer = torch.optim.Adam(agent.nn.parameters(), lr=1e-4)

    def train(self, num_samples: int = 1000) -> None:
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            while num_samples > 0:
                trajectory = self.rollout_trajectory()
                trajectories.append(trajectory)
                n = len(trajectory)
                num_samples -= n
                pbar.update(n)
        self._update_nn(trajectories)

    def rollout_trajectory(self) -> list[MCTSExperience]:
        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        while not board.board_info.game_over and board.board_info.duration < agent.max_game_length:
            move = agent.choose_move(board, training=True)
            board = board.perform_move(move)

        trajectory = agent.trajectory
        for i, experience in enumerate(reversed(trajectory)):
            experience.v_target = -self._gamma**i  # TODO: think about this

        self._agent.on_game_end()
        return trajectory

    def _update_nn(self, trajectories: list[list[MCTSExperience]]) -> None:
        dataset = TrainingDataset(trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )

        for x, value_target, pi_target, pi_mask in tqdm(dataloader, desc="nn update"):
            self._optimizer.zero_grad()
            current_pi, current_value = self._agent.nn(x)
            value_loss = torch.nn.functional.mse_loss(current_value, value_target)
            policy_loss_unmasked = -pi_target * torch.log(current_pi)
            policy_loss = torch.where(pi_mask, policy_loss_unmasked, 0.0)
            loss = (value_loss + policy_loss).mean()
            loss.backward()
            self._optimizer.step()
        self._agent.nn.clear_cache()
