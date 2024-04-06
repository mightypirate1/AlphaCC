from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.engine import Board


def test_integration() -> None:
    def get_winner(starting_player: int) -> int:
        board = Board(board_size).reset_with_starting_player(starting_player)
        agent = GreedyAgent(board_size)
        for _ in range(expected_number_of_moves):
            action = agent.choose_move(board)
            board = board.perform_move(action)
            if board.board_info.game_over:
                break
        assert board.board_info.game_over, "game longer than it should"
        return board.board_info.winner

    board_size = 9
    expected_number_of_moves = 115  # chosen since this was the actual value to complete a game
    first_winner = get_winner(1)
    second_winner = get_winner(2)
    assert first_winner == 1
    assert second_winner == 2
