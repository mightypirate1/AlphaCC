import logging


def init_rootlogger(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    if verbose:
        logging.root.setLevel(logging.DEBUG)
