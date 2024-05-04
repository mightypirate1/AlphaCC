import logging
from typing import Any

import dill
import redis

from alpha_cc.agents.mcts import MCTSExperience

logger = logging.getLogger(__file__)


class TrainingDB:
    def __init__(self, host: str = "localhost") -> None:
        self._db = redis.Redis(host=host)

    @property
    def queue_key(self) -> str:
        return "experience-queue"

    @property
    def latest_weights_key(self) -> str:
        return "weights-latest"

    @property
    def latest_weights_index_key(self) -> str:
        return "latest-weights-index"

    def weight_key(self, index: int | str) -> str:
        return f"weights-{str(index).zfill(4)}"

    def reset_current_weights(self) -> None:
        logger.debug("reseting weights")
        self._db.set(self.latest_weights_index_key, 0)

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

    def weights_is_latest(self, current_index: int) -> bool:
        latest_index = self.fetch_latest_weight_index()
        is_latest = latest_index <= current_index
        logger.debug(f"weights {current_index} is latest: {is_latest}")
        return is_latest

    def first_weights_published(self) -> bool:
        response = self._db.get(self.latest_weights_index_key)
        is_published = response is not None
        logger.debug(f"first weights published: {is_published}")
        return is_published
