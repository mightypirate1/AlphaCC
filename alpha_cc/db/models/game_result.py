from dataclasses import dataclass


@dataclass(frozen=True)
class GameResult:
    channel_p1: int  # channel that played as player 1 (first mover)
    channel_p2: int  # channel that played as player 2
    winner: int  # 0 = draw/timeout, 1 = p1 won, 2 = p2 won
    game_length: int  # number of moves (board.info.duration)
    hit_max_duration: bool  # game was stopped by max_game_length cap
