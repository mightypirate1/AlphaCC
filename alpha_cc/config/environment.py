import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

DOTENV_FILE = Path(__file__).parents[2] / ".env"
load_dotenv(DOTENV_FILE)


@dataclass
class Environment:
    host_redis: ClassVar[str] = os.environ.get("HOST_REDIS", "localhost")
    tb_logdir: ClassVar[str] = os.environ.get("TB_LOGDIR", "data/logdir")
    model_dir: ClassVar[str] = os.environ.get("MODELDIR", "data/models")
