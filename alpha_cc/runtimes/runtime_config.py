from dataclasses import dataclass


@dataclass
class RunTimeConfig:
    verbose: bool = False
    render: bool = False
    slow: bool = False  # Does nothing if render is False
