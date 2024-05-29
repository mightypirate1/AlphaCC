import logging
from typing import Any

import dill
import redis

from alpha_cc.agents.mcts import MCTSExperience

logger = logging.getLogger(__file__)


class TrainingDB:
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

    def weight_key(self, index: int | str) -> str:
        return f"weights-{str(index).zfill(4)}"

    def flush_db(self) -> None:
        logger.debug("reseting db")
        self._db.flushdb()

    def post_trajectory(self, trajectory: list[MCTSExperience]) -> None:
        logger.debug(f"posting {len(trajectory)} experiences")
        self._db.lpush(self.queue_key, dill.dumps(trajectory))

    def fetch_all_trajectories(self) -> list[list[MCTSExperience]]:
        trajectories = []
        while trajectory := self.fetch_trajectory(blocking=False):
            trajectories.append(trajectory)
        return trajectories

    def fetch_trajectory(self, blocking: bool = False) -> list[MCTSExperience]:
        if blocking:
            _, encoded_experiences = self._db.brpop(self.queue_key, timeout=0)  # type: ignore
        else:
            encoded_experiences = self._db.rpop(self.queue_key)
            if encoded_experiences is None:
                return []
        traj = dill.loads(encoded_experiences)  # noqa
        logger.debug(f"fetched {len(traj)} experiences")
        return traj

    def set_current_model(self, channel: int, weight_index: int) -> None:
        self._db.hset(self.current_models_key, key=str(channel), value=str(weight_index))

    def remove_model(self, channel: int) -> None:
        self._db.hdel(self.current_models_key, str(channel))

    def get_current_models(self) -> dict[int, int]:
        channels = self._db.hkeys(self.current_models_key)
        if channels is None:
            return {}
        return {
            int(channel): int(self._db.hget(self.current_models_key, channel)) for channel in channels  # type: ignore
        }

    def publish_latest_weights(self, state_dict: dict[str, Any]) -> int:
        payload = dill.dumps(state_dict)
        self._db.set(self.latest_weights_key, payload)
        current_index = int(self._db.incr(self.latest_weights_index_key))  # type: ignore
        self._db.set(self.weight_key(str(current_index)), payload)
        logger.debug(f"published weights {current_index}")
        return current_index

    def fetch_latest_weights_with_index(self) -> tuple[int, dict[str, Any]]:
        return self.fetch_latest_weight_index(), self.fetch_latest_weights()

    def fetch_latest_weights(self) -> dict[str, Any]:
        logger.debug("fetching latest weights")
        encoded_weights = self._db.get(self.latest_weights_key)
        return dill.loads(encoded_weights)  # noqa

    def fetch_weights(self, index: int | str) -> dict[str, Any]:
        logger.debug(f"fetching weights {index}")
        encoded_weights = self._db.get(self.weight_key(index))
        return dill.loads(encoded_weights)  # noqa

    def fetch_latest_weight_index(self) -> int:
        response = self._db.get(self.latest_weights_index_key)
        index = 0 if response is None else int(response)  # type: ignore
        logger.debug(f"latest index: {index}")
        return index
