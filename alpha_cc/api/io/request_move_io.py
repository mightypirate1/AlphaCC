from alpha_cc.api.io.base_io import BaseIO


class RequestMoveIO(BaseIO):
    game_id: str
    n_rollouts: int = 500
    rollout_depth: int = 100
    temperature: float = 1.0
