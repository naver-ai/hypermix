__all__ = ["retry", "MaxRetryError", "set_seed", "generate_seeds"]

import time
import random
import logging

import numpy as np
import torch


class MaxRetryError(Exception):
    pass


def retry(f, max_retries=10, exc_cls=Exception, desc=None, error_handler=None):
    error = None
    for _ in range(max_retries):
        try:
            return f()
        except exc_cls as e:
            if error_handler is not None:
                error_handler(e)
            error = e
    raise MaxRetryError(f"{desc} failed after {max_retries} retries. "
                        f"last error msg: {error}")


def set_seed(seed=None, purpose=None):
    seed = seed or (time.time_ns() % 2 ** 32)
    if purpose is None:
        logging.info(f"setting seed: {seed}")
    else:
        logging.info(f"setting seed for {purpose}: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def generate_seeds(master_seed=None, purpose=None):
    if purpose is None:
        master_seed_purpose = "generating children seeds"
    else:
        master_seed_purpose = f"generating children seeds for {purpose}"
    set_seed(master_seed, purpose=master_seed_purpose)

    while True:
        yield random.randint(0, 2 ** 32 - 1)
