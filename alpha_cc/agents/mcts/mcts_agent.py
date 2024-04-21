import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.mcts_node import MCTSNode
from alpha_cc.agents.state import GameState, StateHash
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet


class MCTSAgent(Agent):
    def __init__(
        self,
        board_size: int,
        n_rollouts: int = 100,
        rollout_depth: int = 999,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = 0.0  # 0.25  # TODO: get good value from paper
        self._summary_writer = summary_writer
        self._global_step = 0
        self._nn = DefaultNet(board_size)
        self._trajectory: list[MCTSExperience] = []
        self._nodes: dict[StateHash, MCTSNode] = {}
        # stats
        self._rollout_count_total = 0
        self._rollout_count_game_over = 0
        self._rollout_count_recursion_depth = 0
        self._rollout_count_disallowed_children = 0
        self._rollout_count_new_node = 0

    @property
    def trajectory(self) -> list[MCTSExperience]:
        return self._trajectory

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def nodes(self) -> dict[StateHash, MCTSNode]:
        return self._nodes

    def on_game_start(self) -> None:
        self._nodes.clear()
        self._trajectory = []
        self._reset_stats_counters()

    def on_game_end(self) -> None:
        if self._summary_writer is not None:
            self._summary_writer.add_scalar(
                "game-length",
                len(self._trajectory),
                global_step=self._global_step,
            )
            self._summary_writer.add_scalar(
                "rollout/frac-game-over",
                self._rollout_count_game_over / self._rollout_count_total,
                global_step=self._global_step,
            )
            self._summary_writer.add_scalar(
                "rollout/frac-recursion-depth",
                self._rollout_count_recursion_depth / self._rollout_count_total,
                global_step=self._global_step,
            )
            self._summary_writer.add_scalar(
                "rollout/frac-disallowed-children",
                self._rollout_count_disallowed_children / self._rollout_count_total,
                global_step=self._global_step,
            )
            self._summary_writer.add_scalar(
                "rollout/frac-new-node",
                self._rollout_count_new_node / self._rollout_count_total,
                global_step=self._global_step,
            )

    def choose_move(self, board: Board, training: bool = False) -> int | np.integer:
        def _training_policy(pi: np.ndarray) -> np.ndarray:
            if self._dirichlet_weight == 0:
                return pi
            # TODO: learn about this!
            dirichlet_noise = np.random.dirichlet(np.full_like(pi, 1 / len(pi)))
            return (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise

        self._global_step += 1
        traversed_states = {exp.state.hash for exp in self.trajectory}
        state = GameState(board, disallowed_children=traversed_states)
        for _ in range(self._n_rollouts):
            self._rollout_count_total += 1
            v = self._rollout(state, remaining_depth=self._rollout_depth)
        pi = self.pi(state)

        if training:
            experience = MCTSExperience(state=state, pi_target=pi, v_target=v)
            self._trajectory.append(experience)
            pi = _training_policy(pi)
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def pi(self, state: GameState, temperature: float = 1.0) -> np.ndarray:
        node = self._nodes[state.hash]
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        normalized_weighted_counts = weighted_counts / weighted_counts.sum()

        if self._summary_writer is not None:
            self._summary_writer.add_scalar(
                "agent/action-entropy", entropy(normalized_weighted_counts), global_step=self._global_step
            )

        return normalized_weighted_counts

    @torch.no_grad()
    def _rollout(self, state: GameState, remaining_depth: int = 999) -> float:
        """
        recursive rollout that traverses the game-tree and updates nodes along the way.
        the return value is the value as seen by the parent node, hence all the minuses.
        """

        def add_as_new_node(v_hat: float, pi: np.ndarray) -> None:
            self._nodes[state.hash] = MCTSNode(
                v_hat=v_hat,
                pi=pi,
                n=np.zeros(len(state.children), dtype=np.integer),
                q=np.zeros(len(state.children)),
            )

        # if game is over, we stop
        if state.info.game_over:
            self._rollout_count_game_over += 1
            if state.info.winner == state.info.current_player:
                return 1.0  # previous player won
            return -1.0

        # at some point one has to stop (recursion limit, feasability, etc)
        if remaining_depth == 0:
            self._rollout_count_recursion_depth += 1
            return -float(self._nn.value(state))

        # we also don't want to go around in circles
        if all(sp.hash in state.disallowed_children for sp in state.children):
            self._rollout_count_disallowed_children += 1
            return -float(self._nn.value(state))

        # if we have reached as far as we have been:
        # - initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes:
            self._rollout_count_new_node += 1
            v_hat = float(self._nn.value(state))
            pi = self._nn.policy(state)
            add_as_new_node(v_hat, pi)
            return -v_hat

        # keep rolling
        a = self._find_best_action(state)
        s_prime = GameState(
            state.board.perform_move(a),
            disallowed_children={state.hash, *state.disallowed_children},
        )
        v = self._rollout(s_prime, remaining_depth=remaining_depth - 1)

        # update node
        node = self._nodes[state.hash]
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1

        return -v

    def _find_best_action(self, state: GameState) -> int:
        def c_puct(node: MCTSNode) -> float:  # TODO: look at this again
            # according to some paper i forgot to reference...
            c_puct_init = 2.5
            c_puct_base = 19652
            return c_puct_init + np.log((node.n.sum() + c_puct_base + 1) / c_puct_base)

        best_u, best_a = -np.inf, -1
        node = self._nodes[state.hash]
        sum_n = sum(node.n)

        for a, sp in enumerate(state.children):
            if sp.hash in state.disallowed_children:
                continue
            q_sa = node.q[a]
            p_sa = node.pi[a]
            u = q_sa + c_puct(node) * p_sa * np.sqrt(sum_n) / (1 + node.n[a])

            if u > best_u:
                best_u = u
                best_a = a

        if best_a == -1:
            raise ValueError("our disallowed children got us stuck?")

        return best_a

    def _reset_stats_counters(self) -> None:
        self._rollout_count_total = 0
        self._rollout_count_game_over = 0
        self._rollout_count_recursion_depth = 0
        self._rollout_count_disallowed_children = 0
        self._rollout_count_new_node = 0
