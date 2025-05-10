import time

import click

from alpha_cc.config import Environment
from alpha_cc.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.nn.service import NNService


@click.command("alpha-cc-nn-service")
@click.option("--size", type=int, default=9)
@click.option("--reload-frequency", type=int, default=1)
@click.option("--gpu", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(size: int, reload_frequency: int, gpu: bool, verbose: bool) -> None:
    init_rootlogger(verbose=verbose)
    # let the trainer start and flush the db
    time.sleep(5)

    nn_service = NNService(
        nn_creator=lambda: DefaultNet(size),
        host=Environment.redis_host,
        reload_frequency=reload_frequency,
        gpu=gpu,
    )
    nn_service.run()
