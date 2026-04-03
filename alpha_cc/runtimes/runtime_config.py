from dataclasses import dataclass


@dataclass
class RunTimeConfig:
    max_game_length: int | None = None
    verbose: bool = False
    render: bool = False
    slow: bool = False  # Does nothing if render is False
