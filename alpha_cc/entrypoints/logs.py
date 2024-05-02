import logging


def init_rootlogger(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    logging.root.addHandler(handler)
    if verbose:
        logging.root.setLevel(logging.INFO)
