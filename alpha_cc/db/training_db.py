from typing import OrderedDict
import dill
import redis
import torch

from alpha_cc.agents.mcts import MCTSExperience


class TrainingDB:
    def __init__(self, host: str = "localhost") -> None:
        self._db = redis.Redis(host=host)
        
    @property
    def queue_key(self) -> str:
        return "experience-queue"
    
    @property
    def latest_weights_key(self) -> str:
        return "latest-weights"

    @property
    def latest_weights_index_key(self) -> str:
        return "latest-weights-index"
        
    def post_experiences(self, experiences: list[MCTSExperience]) -> None:
        self._db.lpush(self.queue_key, dill.dumps(experiences))
        
    def fetch_experiences(self, blocking: bool = False) -> list[MCTSExperience]:
        if blocking:
            _, encoded_experiences = self._db.brpop(self.queue_key, timeout=0)
        else:
            encoded_experiences = self._db.rpop(self.queue_key)
            if encoded_experiences is None:
                return []
        return dill.loads(encoded_experiences)
    
    def publish_latest_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        self._db.set(self.latest_weights_key, dill.dumps(state_dict))
        self._db.incr(self.latest_weights_index_key)
        
    def fetch_latest_weights(self) -> OrderedDict[str, torch.Tensor]:
        encoded_weights = self._db.get(self.latest_weights_key)
        return dill.loads(encoded_weights)
    
    def weights_is_latest(self, current_index: int) -> bool:
        response = self._db.get(self.latest_weights_index_key)
        if response is None:
            return True  # if we don't have any weights, we don't tell people to fetch
        return int(response) > current_index

