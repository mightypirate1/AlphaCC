import pytest

from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.engine import Board

from .common import get_random_board_state


def test_integration() -> None:
    def get_winner(starting_player: int) -> int:
        move_count = 0
        board = Board(board_size).reset_with_starting_player(starting_player)
        agent = GreedyAgent(board_size)
        while not board.board_info.game_over:
            move_count += 1
            assert move_count < 1000, "game seems infinite"
            action = agent.choose_move(board)
            board = board.perform_move(action)
        assert move_count == expected_number_of_moves, "incorrect number of moves"
        assert board.board_info.game_over, "game not over"
        return board.board_info.winner

    board_size = 9
    expected_number_of_moves = 122  # chosen since this was the actual value to complete a game
    first_winner = get_winner(1)
    second_winner = get_winner(2)
    assert first_winner == 2
    assert second_winner == 1


@pytest.mark.parametrize("board_size", range(3, 13))
def test_moves(board_size: int) -> None:
    board = get_random_board_state(board_size)
    moves = board.get_legal_moves()
    move_mask = moves.get_action_mask()
    action_mask_indices = moves.get_action_mask_indices()

    masked_moves = set()
    for i, _ in enumerate(moves):
        from_coord, to_coord = action_mask_indices[i]
        assert move_mask[from_coord.x][from_coord.y][to_coord.x][to_coord.y]
        masked_moves.add((from_coord.x, from_coord.y, to_coord.x, to_coord.y))

    for i in range(board_size):
        for j in range(board_size):
            for k in range(board_size):
                for l in range(board_size):  # noqa
                    if (i, j, k, l) not in masked_moves:
                        assert not move_mask[i][j][k][l]
