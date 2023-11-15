import json
import multiprocessing as mp
import os
import random
from typing import Union, Callable

import numpy as np
import torch as th

import wandb
from .optical_flow import OpticalFlow



def init_wandb(args):
    if os.path.exists("../keys.json"):
       with open("../keys.json") as f:
         os.environ["WANDB_API_KEY"] = json.load(f)["api_key"]

    wandb.tensorboard.patch(root_logdir="./runs", pytorch=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="ppo_lstm",
        sync_tensorboard = True,
        name = args.NAME,
        # track hyperparameters and run metadata
        config=args
    )


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def exp_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return (progress_remaining)**2 * initial_value

    return func


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def optical_flow_create_shared_vars():
    manager = mp.Manager()
    # queue = multiprocessing.Queue()
    shared_dict = manager.dict()
    shared_queue = manager.Queue()
    shared_var = (shared_queue, shared_dict)
    if os.name == "posix":
        ctx = mp.get_context("forkserver")
    else:
        ctx = mp.get_context("spawn")
    process = ctx.Process(target=OpticalFlow, args=((224, 224), True, shared_var),
                          daemon=True)  # type: ignore[attr-defined]
    # pytype: enable=attribute-error
    process.start()
    return shared_var
