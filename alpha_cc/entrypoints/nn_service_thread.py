import time

import click

from alpha_cc.config import Environment
from alpha_cc.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.nn.service import NNService


@click.command("alpha-cc-nn-service")
@click.option("--size", type=int, default=9)
@click.option("--reload-frequency", type=int, default=5)
@click.option("--log-frequency", type=int, default=60)
@click.option("--inference-batch-size", type=int, default=512)
@click.option("--num-post-workers", type=int, default=2)
@click.option("--gpu", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    size: int,
    reload_frequency: int,
    log_frequency: int,
    inference_batch_size: int,
    num_post_workers: int,
    gpu: bool,
    verbose: bool,
) -> None:
    init_rootlogger(verbose=verbose)
    # let the trainer start and flush the db
    time.sleep(5)

    nn_service = NNService(
        nn_creator=lambda: DefaultNet(size),
        redis_host_main=Environment.redis_host_main,
        redis_host_pred=Environment.redis_host_pred,
        reload_frequency=reload_frequency,
        log_frequency=log_frequency,
        infecence_batch_size=inference_batch_size,
        num_post_workers=num_post_workers,
        gpu=gpu,
    )
    nn_service.run()
