import alpha_cc
import numpy as np

board = alpha_cc.Board(9)
print(board)
print(board.reset())
board.render()
print("winner", board.get_board_info().winner)
# exit("!")
# for i in range(1, 1000):
#     x = board.get_all_possible_next_states()
#     move_idx = np.random.choice([i for i,_ in enumerate(x)])
#     board = board.perform_move(move_idx)
#     # print(f"after move {move_idx}")
#
# board.render()
# print(board.get_matrix())
# print(board.get_matrix())
# print(id(board.get_matrix()))
# print(id(board.get_matrix()))
# print(id(board.get_matrix()))
