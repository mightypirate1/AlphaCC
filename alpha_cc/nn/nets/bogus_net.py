# import numpy as np

# from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
# from alpha_cc.state import GameState


# class BogusNet:
#     def __init__(self, board_size: int) -> None:
#         self._heuristic = HeuristicReward(board_size)

#     def policy(self, state: GameState) -> np.ndarray:
#         pi = np.array([self.heuristic(sp) for sp in state.children])
#         pi = pi / pi.sum()
#         return pi

#     def value(self, state: GameState) -> np.floating:
#         return np.float32(self.heuristic(state))

#     def update_weights(self, train_data: list[list[MCTSExperience]]) -> None:
#         print(f"pretend training on {len(train_data)} trajectories")
