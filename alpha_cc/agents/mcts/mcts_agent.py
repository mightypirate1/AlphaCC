import numpy as np
import torch

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.heuristic import Heuristic
from alpha_cc.agents.mcts.mcts_node import MCTSNode
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.state import GameState, StateHash


class MCTSAgent(Agent):
    def __init__(
        self,
        board_size: int,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        dirichlet_weight: float = 0.0,  # TODO: get good value from paper
    ) -> None:
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = dirichlet_weight
        self._nn = DefaultNet(board_size)
        self._heuristic = Heuristic(board_size, subtract_opponent=True)
        self._nodes: dict[StateHash, MCTSNode] = {}

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def nodes(self) -> dict[StateHash, MCTSNode]:
        return self._nodes

    def on_game_start(self) -> None:
        self.nodes.clear()
        self.nn.clear_cache()

    def on_game_end(self) -> None:
        pass

    def choose_move(self, board: Board, training: bool = False, temperature: float = 1.0) -> int | np.integer:
        pi, _ = self.run_rollouts(board, training=training, temperature=temperature)
        if training:
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def run_rollouts(self, board: Board, training: bool = False, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        state = GameState(board)
        value = np.array(
            [-self._rollout(state, remaining_depth=self._rollout_depth) for _ in range(self._n_rollouts)]
        ).mean()
        pi = self._rollout_policy(state, training, temperature)
        return pi, value

    def _rollout_policy(self, state: GameState, training: bool = False, temperature: float = 1.0) -> np.ndarray:
        node = self._nodes[state.hash]
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        pi = weighted_counts / weighted_counts.sum()
        if not training:
            return pi
        dirichlet_noise = np.random.dirichlet(np.full_like(pi, 1 / len(pi)))
        pi_noised = (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise
        return pi_noised

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
                moves=state.moves,
            )

        # if game is over, we stop
        if state.info.game_over:
            if state.info.winner == state.info.current_player:
                return -1.0  # previous player lost
            return 1.0  # previous player won

        # if we have reached as far as we have been:
        # - initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes:
            v_hat = float(self.nn.value(state))
            pi = self.nn.policy(state)
            add_as_new_node(v_hat, pi)
            return -v_hat

        # prepare continued rollout
        node = self._nodes[state.hash]
        a = self._find_best_action(state)

        # at some point one has to stop (recursion limit, feasability, etc)
        if remaining_depth == 0:
            v = float(self.nn.value(state))
            return -v

        s_prime = GameState(state.board.apply(node.moves[a]))
        v = self._rollout(s_prime, remaining_depth=remaining_depth - 1)

        # update node
        node = self._nodes[state.hash]
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1

        return -v

    def _find_best_action(self, state: GameState) -> int:
        def node_c_puct() -> float:  # TODO: look at this again
            # according to some paper i forgot to reference...
            c_puct_init = 2.5
            c_puct_base = 19652
            return c_puct_init + np.log((node.n.sum() + c_puct_base + 1) / c_puct_base)

        node = self._nodes[state.hash]
        sum_n = sum(node.n)
        c_puct = node_c_puct()

        u = node.q + c_puct * node.pi * np.sqrt(sum_n) / (1 + node.n)
        best_action = np.argmax(u).astype(int)

        return best_action
