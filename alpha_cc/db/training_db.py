import logging
from typing import Any

import dill
import redis

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.db.models.tournament_results import TournamentResult

logger = logging.getLogger(__file__)


class TrainingDB:
    """
    Redis database for training data.

    Data categories:
    - trajectory
    - model
    - weights
    - tournament
    """

    def __init__(self, host: str = "localhost") -> None:
        self._db = redis.Redis(host=host, db=0)

    @property
    def queue_key(self) -> str:
        return "experience-queue"

    @property
    def latest_weights_key(self) -> str:
        return "weights-latest"

    @property
    def current_models_key(self) -> str:
        return "update-models"

    @property
    def latest_weights_index_key(self) -> str:
        return "latest-weights-index"

    @property
    def tournament_counter_key(self) -> str:
        return "tournament-counter"

    @property
    def tournament_results_key(self) -> str:
        return "tournament-results"

    @property
    def tournament_queue_key(self) -> str:
        return "tournament-queue"

    def weight_key(self, index: int | str) -> str:
        return f"weights-{str(index).zfill(4)}"

    def tournament_result_key(self, channel_1: int, channel_2: int, winner: int) -> str:
        return f"{channel_1}-{channel_2}-{winner}"

    def flush_db(self) -> None:
        logger.debug("reseting db")
        self._db.flushdb()

    ##
    # trajectory
    def trajectory_post(self, trajectory: list[MCTSExperience]) -> None:
        logger.debug(f"posting {len(trajectory)} experiences")
        self._db.lpush(self.queue_key, dill.dumps(trajectory))

    def trajectory_fetch_all(self) -> list[list[MCTSExperience]]:
        trajectories = []
        while trajectory := self.trajectory_fetch(blocking=False):
            trajectories.append(trajectory)
        return trajectories

    def trajectory_fetch(self, blocking: bool = False) -> list[MCTSExperience]:
        if blocking:
            _, encoded_experiences = self._db.brpop(self.queue_key, timeout=0)  # type: ignore
        else:
            encoded_experiences = self._db.rpop(self.queue_key)
            if encoded_experiences is None:
                return []
        traj = dill.loads(encoded_experiences)  # noqa
        logger.debug(f"fetched {len(traj)} experiences")
        return traj

    ##
    # model
    def model_set_current(self, channel: int, weight_index: int) -> None:
        self._db.hset(self.current_models_key, key=str(channel), value=str(weight_index))

    def model_remove(self, channel: int) -> None:
        self._db.hdel(self.current_models_key, str(channel))

    def model_get_current(self) -> dict[int, int]:
        channels = self._db.hkeys(self.current_models_key)
        if channels is None:
            return {}
        return {
            int(channel): int(self._db.hget(self.current_models_key, channel)) for channel in channels  # type: ignore
        }

    ##
    # weights
    def weights_publish_latest(self, state_dict: dict[str, Any]) -> int:
        current_index = int(self._db.incr(self.latest_weights_index_key))  # type: ignore
        self.weights_publish(state_dict, current_index, set_latest=True)
        return current_index

    def weights_publish(self, state_dict: dict[str, Any], index: int, set_latest: bool = False) -> None:
        payload = dill.dumps(state_dict)
        self._db.set(self.weight_key(index), payload)
        if set_latest:
            self._db.set(self.latest_weights_key, payload)
        logger.debug(f"published weights {index}")

    def weights_fetch_latest_with_index(self) -> tuple[int, dict[str, Any]]:
        return self.weights_fetch_latest_index(), self.weights_fetch_latest()

    def weights_fetch_latest(self) -> dict[str, Any]:
        logger.debug("fetching latest weights")
        encoded_weights = self._db.get(self.latest_weights_key)
        return dill.loads(encoded_weights)  # noqa

    def weights_fetch(self, index: int | str) -> dict[str, Any]:
        logger.debug(f"fetching weights {index}")
        encoded_weights = self._db.get(self.weight_key(index))
        return dill.loads(encoded_weights)  # noqa

    def weights_fetch_latest_index(self) -> int:
        response = self._db.get(self.latest_weights_index_key)
        index = 0 if response is None else int(response)  # type: ignore
        logger.debug(f"latest index: {index}")
        return index

    ##
    # tournament
    def tournament_reset(self) -> None:
        self._db.set(self.tournament_counter_key, 0)
        self._db.delete(self.tournament_queue_key)
        self._db.delete(self.tournament_results_key)

    def tournament_increment_counter(self) -> None:
        self._db.incr(self.tournament_counter_key, 1)

    def tournament_add_match(self, channel_1: int, channel_2: int) -> None:
        paring = (channel_1, channel_2)
        self._db.lpush(self.tournament_queue_key, dill.dumps(paring))

    def tournament_get_match(self) -> tuple[int, int] | None:
        encoded_paring = self._db.rpop(self.tournament_queue_key)
        if encoded_paring is None:
            return None
        return dill.loads(encoded_paring)  # noqa

    def tournament_get_n_completed_games(self) -> int:
        return int(self._db.get(self.tournament_counter_key))  # type: ignore

    def tournament_add_result(self, channel_1: int, channel_2: int, winner: int) -> None:
        pairing_key = self.tournament_result_key(channel_1, channel_2, winner)
        self._db.hincrby(self.tournament_results_key, pairing_key, 1)

    def tournament_get_results(self) -> TournamentResult:
        return TournamentResult.from_db(self.tournament_results_key, self._db)
