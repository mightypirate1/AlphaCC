import pandas as pd
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import Board
from alpha_cc.runtimes.runtime import RunTime


class TournamentRuntime:
    def __init__(self, board: Board, agents: dict[str, Agent]) -> None:
        self._board = board
        self._agents = agents

    def run_tournament(self, num_iterations: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self._evaluate_all_pairings(player_1_key, player_1, num_iterations)
                for player_1_key, player_1 in tqdm(self._agents.items(), desc="tournament")
            ]
        )

    def _evaluate_all_pairings(self, player_1_key: str, player_1: Agent, num_iterations: int) -> pd.DataFrame:
        return pd.Series(
            {
                player_2_key: self._evaluate_pairing(player_1, player_2, num_iterations)
                for player_2_key, player_2 in self._agents.items()
            },
            name=player_1_key,
        )

    def _evaluate_pairing(self, player_1: Agent, player_2: Agent, num_iterations: int) -> float:
        runtime = RunTime(self._board, (player_1, player_2))
        n_wins = 0
        for _ in range(num_iterations):
            winner = runtime.play_game()
            if winner == 1:
                n_wins += 1
        return n_wins / num_iterations
