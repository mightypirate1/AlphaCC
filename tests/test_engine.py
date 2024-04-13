import pytest

from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.engine import Board


def test_integration() -> None:
    def get_winner(starting_player: int) -> int:
        move_count = 0
        board = Board(board_size).reset_with_starting_player(starting_player)
        agent = GreedyAgent(board_size)
        while not board.board_info.game_over:
            if (move_count := move_count + 1) > 1000:
                assert "game seems infinite"
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


@pytest.mark.parametrize("board_size", range(13))
def test_moves(board_size: int) -> None:
    board = Board(board_size)
    moves = board.get_legal_moves()
    move_mask = moves.get_action_mask()
    action_mask_indices = moves.get_action_mask_indices()

    for i, _ in enumerate(moves):
        from_coord, to_coord = action_mask_indices[i]
        assert move_mask[from_coord.x][from_coord.y][to_coord.x][to_coord.y]
    # TODO: also test that entries in the mask that are False are NOT in the action_mask_indices
