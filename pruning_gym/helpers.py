import json
import multiprocessing as mp
import os
import random
from typing import Union, Callable

import numpy as np
import torch as th

import wandb
from nptyping import NDArray, Shape, Float

from .optical_flow import OpticalFlow
import time


def init_wandb(args, name):
    if os.path.exists("../keys.json"):
       with open("../keys.json") as f:
         os.environ["WANDB_API_KEY"] = json.load(f)["api_key"]

    wandb.tensorboard.patch(root_logdir="runs", pytorch=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="ppo_lstm",
        sync_tensorboard = True,
        name = name,
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


def optical_flow_create_shared_vars(num_envs: int = 1):
    #TODO = make this torch multiprocessing
    manager = mp.Manager()

    # queue = multiprocessing.Queue()
    shared_dict = manager.dict()
    shared_queue = manager.Queue()
    shared_var = (shared_queue, shared_dict)
    if os.name == "posix":
        ctx = mp.get_context("forkserver")
    else:
        ctx = mp.get_context("spawn")
    #replace shared dict and queue with pipe?
    process = ctx.Process(target=OpticalFlow, args=((224, 224), True, shared_var, num_envs),
                          daemon=True)  # type: ignore[attr-defined]
    # pytype: enable=attribute-error
    process.start()
    time.sleep(1)
    return shared_var


def compute_perpendicular_projection_vector(ab: NDArray[Shape['3, 1'], Float], bc: NDArray[Shape['3, 1'], Float]):
    projection = ab - np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection



def goal_distance(goal_a: NDArray[Shape['3, 1'], Float], goal_b: NDArray[Shape['3, 1'], Float]) -> float:
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)




def goal_reward_projection(current: NDArray[Shape['3, 1'], Float], previous: NDArray[Shape['3, 1'], Float],
                           target: NDArray[Shape['3, 1'], Float]):
    # Compute the reward between the previous and current goal.
    assert current.shape == previous.shape
    assert current.shape == target.shape
    # get parallel projection of current on prev
    projection = compute_parallel_projection_vector(current - target, previous - target)

    reward = np.linalg.norm(previous - target) - np.linalg.norm(projection)
    # print(np.linalg.norm(previous - target), np.linalg.norm(projection), np.linalg.norm(current - target))

    return reward


# x,y distance
def goal_distance2d(goal_a: NDArray[Shape['3, 1'], Float], goal_b: NDArray[Shape['3, 1'], Float]):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

def compute_parallel_projection_vector(ab: NDArray[Shape['3, 1'], Float], bc: NDArray[Shape['3, 1'], Float]):
    projection = np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection

def set_args(arg_dict, parser, name = 'main'):
     for arg_name, arg_params in arg_dict.items():
        # print(arg_name, arg_params)
        if 'args' in arg_name:
            set_args(arg_params, parser, arg_name)
        else:
            parser.add_argument(f'--{name}_{arg_name}', **arg_params)
    # return parser_dict

def organize_args(args_dict):
    parse_args_dict = {}
    args_classes = ['args_global', 'args_train', 'args_test', 'args_record', 'args_callback', 'args_policy', 'args_env', 'args_eval']
    for arg_name in args_classes:
        parse_args_dict[arg_name] = {}
    for arg_name, arg_params in args_dict.items():
        index = arg_name.index('_', 5)
        arg_val = arg_name[index+1:]
        arg_key = arg_name[:index]
        parse_args_dict[arg_key][arg_val] = arg_params
    return parse_args_dict


def add_arg_to_env(key, val, env_name, parsed_args_dict):
    for name in env_name:
        parsed_args_dict[name][key] = val



