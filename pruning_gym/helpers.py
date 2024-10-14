import json
import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Union, Callable

import numpy as np
import torch as th
import wandb
from nptyping import NDArray, Shape, Float
import re
import ast
from .optical_flow import OpticalFlow
import math

def init_wandb(args, name):
    if os.path.exists("../keys.json"):
        with open("../keys.json") as f:
            os.environ["WANDB_API_KEY"] = json.load(f)["api_key"]

    wandb.tensorboard.patch(root_logdir="runs", pytorch=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="ppo_lstm",
        sync_tensorboard=True,
        name=name,
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
        return (progress_remaining) ** 2 * initial_value

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


def optical_flow_create_shared_vars(num_envs: int = 1, algo_size=(224, 224)):
    manager = mp.Manager()

    # queue = multiprocessing.Queue()
    shared_dict = manager.dict()
    shared_queue = manager.Queue()
    shared_var = (shared_queue, shared_dict)
    if os.name == "posix":
        ctx = mp.get_context("forkserver")
    else:
        ctx = mp.get_context("spawn")
    # replace shared dict and queue with pipe?
    process = ctx.Process(target=OpticalFlow, args=(algo_size, True, shared_var, num_envs),
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


def compute_parallel_projection_vector(ab: NDArray[Shape['3, 1'], Float], bc: NDArray[Shape['3, 1'], Float]):
    projection = np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection


def set_args(arg_dict, parser, name='main'):
    for arg_name, arg_params in arg_dict.items():
        # print(arg_name, arg_params)
        if 'args' in arg_name:
            set_args(arg_params, parser, arg_name)
        else:
            parser.add_argument(f'--{name}_{arg_name}', **arg_params)


# return parser_dict

def organize_args(args_dict):
    parse_args_dict = {}
    args_classes = ['args_global', 'args_train', 'args_test', 'args_record', 'args_callback', 'args_policy', 'args_env',
                    'args_eval', 'args_baseline']
    for arg_name in args_classes:
        parse_args_dict[arg_name] = {}
    for arg_name, arg_params in args_dict.items():
        index = arg_name.index('_', 5)
        arg_val = arg_name[index + 1:]
        arg_key = arg_name[:index]
        parse_args_dict[arg_key][arg_val] = arg_params
    args_global = parse_args_dict['args_global']
    args_train = dict(parse_args_dict['args_env'], **parse_args_dict['args_train'])
    args_test = dict(parse_args_dict['args_env'], **parse_args_dict['args_test'])
    args_record = dict(args_test, **parse_args_dict['args_record'])
    args_baseline = parse_args_dict['args_baseline']
    args_policy = parse_args_dict['args_policy']
    args_callback = parse_args_dict['args_callback']
    args_env = parse_args_dict['args_env']
    args_eval = parse_args_dict['args_eval']
    return args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parse_args_dict


def add_arg_to_env(key, val, env_name, parsed_args_dict):
    for name in env_name:
        parsed_args_dict[name][key] = val


def make_or_bins(args, type):
    from pruning_sb3.pruning_gym.pruning_env import PruningEnv
    from pruning_sb3.pruning_gym.tree import Tree
    if os.path.exists(f"{type}_or_bins_{args['tree_count']}.pkl"):
        with open(f"{type}_or_bins_{args['tree_count']}.pkl", "rb") as f:
            or_bins = pickle.load(f)
            for key in or_bins.keys():
                random.shuffle(or_bins[key])
    else:
        data_env_train = PruningEnv(**args, make_trees=True)
        or_bins = Tree.create_bins(18, 36)
        for key in or_bins.keys():
            for i in data_env_train.trees:
                or_bins[key].extend(i.or_bins[key])

        del data_env_train
        # Shuffle the data inside the bisn
        for key in or_bins.keys():
            random.shuffle(or_bins[key])

        with open(f"{type}_or_bins_{args['tree_count']}.pkl", "wb") as f:
            pickle.dump(or_bins, f)
    return or_bins


def get_policy_kwargs(args_policy, args_env, features_extractor_class):
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {"features_dim": args_policy['state_dim'],
                                      "in_channels": 3,
                                      "size": (args_env['cam_height'], args_env['cam_width'])},
        "optimizer_class": th.optim.Adam,
        "log_std_init": args_policy['log_std_init'],
        "net_arch": dict(
            qf=[args_policy['emb_size'] * 2, args_policy['emb_size'],
                args_policy['emb_size'] // 2],
            pi=[args_policy['emb_size'] * 2, args_policy['emb_size'],
                args_policy['emb_size'] // 2]),
        "activation_fn": th.nn.ReLU,
        "share_features_extractor": False,
        "n_lstm_layers": 2,
        "features_dim_critic_add": 2,  # Assymetric critic
        "lstm_hidden_size": 128,
        "algo_size": (args_env['algo_height'], args_env['algo_width']),
    }

    return policy_kwargs

def convert_string(data_str):
    # Find all array parts and replace them with placeholders
    array_pattern = re.compile(r'array\((.*?)\)', re.DOTALL)
    arrays = array_pattern.findall(data_str)
    array_map = {}
    for i, array_str in enumerate(arrays):
        placeholder = f'__ARRAY_PLACEHOLDER_{i}__'
        data_str = data_str.replace(f'array({array_str})', f'"{placeholder}"')
        array_str = array_str.replace('\n', '')
        array_map[placeholder] = np.array(ast.literal_eval(array_str))
    # print(data_str)
    # Use ast.literal_eval to evaluate the modified string
    data_list = ast.literal_eval(data_str)

    # Function to replace placeholders with actual NumPy arrays
    def replace_placeholders(item):
        if isinstance(item, str) and item in array_map:
            return array_map[item]
        return item

    # Apply replacement function to each element
    if isinstance(data_list, list):
        data_list = [replace_placeholders(item) for item in data_list]
    elif isinstance(data_list, tuple):
        data_list = tuple(replace_placeholders(item) for item in data_list)

    return data_list
def roundup(x):
    return math.ceil(x / 10.0) * 10

def rounddown(x):
    return math.floor(x / 10.0) * 10
