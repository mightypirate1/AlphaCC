import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

DOTENV_FILE = Path(__file__).parents[2] / ".env"
load_dotenv(DOTENV_FILE)


@dataclass
class Environment:
    redis_host: ClassVar[str] = os.environ.get("REDIS_HOST", "localhost")
    tb_logdir: ClassVar[str] = os.environ.get("TB_LOGDIR", "data/logdir")
    model_dir: ClassVar[str] = os.environ.get("MODELDIR", "data/models/api")
