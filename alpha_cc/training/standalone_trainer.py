import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.agents.state.game_state import GameState
from alpha_cc.engine import Board
from alpha_cc.nn.blocks import PolicyLogSoftmax
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
        batch_size: int = 64,
        gamma: float = 1.0,
        gamma_delay: int | float = np.inf,
        lr: float = 1e-4,
        summary_writer: SummaryWriter | None = None,
        apply_heuristic_on_max_game_length: bool = False,
    ) -> None:
        self._agent = agent
        self._board = board
        self._policy_weight = policy_weight
        self._value_weight = value_weight
        self._epochs_per_update = epochs_per_update
        self._batch_size = batch_size
        self._gamma = gamma
        self._gamma_delay = gamma_delay
        self._apply_heuristic_on_max_game_length = apply_heuristic_on_max_game_length
        self._policy_log_softmax = PolicyLogSoftmax(board.size)
        self._optimizer = torch.optim.Adam(agent.nn.parameters(), lr=lr, weight_decay=1e-4)
        self._heuristic = HeuristicReward(board.size, scale=1, subtract_opponent=True)
        self._global_step = 0
        self._summary_writer = summary_writer

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
        self._report_rollout_stats(trajectories)
        self._update_nn(trajectories)
        self._global_step += 1

    @torch.no_grad()
    def rollout_trajectory(self, max_game_length: int | None = None) -> list[MCTSExperience]:
        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        def game_exceeds_duration() -> bool:
            if max_game_length is None:
                return False
            return board.board_info.duration >= max_game_length

        while not board.board_info.game_over and not game_exceeds_duration():
            move = agent.choose_move(board, training=True)
            board = board.perform_move(move)

        last_player_value = self._value_from_perspective_of_last_player(board)
        trajectory = self._assign_value_targets(agent.trajectory, value=last_player_value)
        self._agent.on_game_end()
        return trajectory

    def _value_from_perspective_of_last_player(self, board: Board) -> float:
        if board.board_info.game_over:
            return -float(board.board_info.reward)  # minus because this board does not make it onto the trajectory
        if self._apply_heuristic_on_max_game_length:
            return self._heuristic(GameState(board))  # no minus, since the last experience has the previous board
        return self._agent.trajectory[-1].v_target

    def _assign_value_targets(self, trajectory: list[MCTSExperience], value: float = 0.0) -> list[MCTSExperience]:
        for backwards_i, experience in enumerate(reversed(trajectory)):
            experience.v_target = value
            value *= -1.0 if backwards_i > self._gamma_delay else -self._gamma
        return trajectory

    def _update_nn(self, trajectories: list[list[MCTSExperience]]) -> None:
        def train_epoch(epoch: int) -> tuple[float, float]:
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            with tqdm(total=len(dataset), desc=f"nn-update/epoch {epoch}") as pbar:
                for x, target_value, target_pi, pi_mask in dataloader:
                    self._optimizer.zero_grad()
                    current_pi_unsoftmaxed, current_value = self._agent.nn(x)
                    value_loss = compute_value_loss(current_value, target_value)
                    policy_loss = compute_policy_loss(current_pi_unsoftmaxed, target_pi, pi_mask)
                    loss = self._value_weight * value_loss + self._policy_weight * policy_loss
                    loss.backward()
                    self._optimizer.step()
                    epoch_value_loss += value_loss.item() / len(dataloader)
                    epoch_policy_loss += policy_loss.item() / len(dataloader)
                    pbar.update(x.shape[0])
                    pbar.set_postfix(
                        {"value-loss": round(epoch_value_loss, 5), "policy-loss": round(epoch_policy_loss, 5)}
                    )
            return epoch_value_loss, epoch_policy_loss

        def compute_value_loss(current_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(current_value, target_value).mean()

        def compute_policy_loss(
            current_pi_tensor_unsoftmaxed: torch.Tensor, target_pi: torch.Tensor, pi_mask: torch.Tensor
        ) -> torch.Tensor:
            policy_loss_unmasked = -target_pi * self._policy_log_softmax(current_pi_tensor_unsoftmaxed, pi_mask)
            policy_loss = torch.where(pi_mask, policy_loss_unmasked, 0).sum() / pi_mask.sum()
            return policy_loss

        self._agent.nn.train()
        self._agent.nn.clear_cache()
        dataset = TrainingDataset(trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

        total_value_loss = 0.0
        total_policy_loss = 0.0
        for epoch in range(1, self._epochs_per_update + 1):
            epoch_value_loss, epoch_policy_loss = train_epoch(epoch)
            total_value_loss += epoch_value_loss / self._epochs_per_update
            total_policy_loss += epoch_policy_loss / self._epochs_per_update
        if self._summary_writer is not None:
            v_targets = np.array([e.v_target for traj in trajectories for e in traj])
            self._summary_writer.add_scalar("trainer/value-loss", total_value_loss, global_step=self._global_step)
            self._summary_writer.add_scalar("trainer/policy-loss", total_policy_loss, global_step=self._global_step)
            self._summary_writer.add_histogram("trainer/v_target", v_targets, global_step=self._global_step)

    def _report_rollout_stats(self, trajectories: list[list[MCTSExperience]]) -> None:
        def log_aggregates(key: str, data: np.ndarray) -> None:
            if self._summary_writer is None:
                return
            self._summary_writer.add_scalar(f"train-rollouts/{key}-mean", data.mean(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"train-rollouts/{key}-min", data.min(), global_step=self._global_step)
            self._summary_writer.add_scalar(f"train-rollouts/{key}-max", data.max(), global_step=self._global_step)

        if self._summary_writer is not None:
            game_lengths = np.array([len(traj) for traj in trajectories])
            final_state_v_targets = np.array([traj[-1].v_target for traj in trajectories])
            final_state_pi_target_entropies = np.array([entropy(traj[-1].pi_target) for traj in trajectories])
            log_aggregates("game-length", game_lengths)
            log_aggregates("final-state-v-targets", final_state_v_targets)
            log_aggregates("final-state-pi-target-entropies", final_state_pi_target_entropies)
