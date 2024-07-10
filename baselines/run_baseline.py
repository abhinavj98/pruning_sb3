import os
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.pruning_env import PruningEnvRRT
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, make_or_bins
import argparse
import pickle
import pandas as pd
from stable_baselines3.common import utils
from pruning_sb3.baselines.baselines_callbacks import PruningRRTSetGoalCallback, GenerateResults
parser = argparse.ArgumentParser()

#TODO: Not tested. Required refactoring
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, parsed_args_dict = organize_args(
        parsed_args)

    print(parsed_args_dict)
    or_bins = make_or_bins(args_train, "train")

    env = make_vec_env(PruningEnv, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    env = make_vec_env(PruningEnvRRT, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)




    dataset = None
    type = "uniform"
    if os.path.exists(f"{type}_dataset.pkl"):
        with open(f"{type}_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    # Shuffle dataset
    # random.shuffle(dataset)
    set_goal_callback = PruningRRTSetGoalCallback(or_bins=or_bins, type=type, dataset=dataset,
                                                   num_orientations=args_callback['n_eval_orientations'],
                                                   num_points_per_or=args_callback['n_points_per_orientation'],
                                                   verbose=args_callback['verbose'])

    if set_goal_callback.dataset is None:
        set_goal_callback.make_dataset()
    random.shuffle(set_goal_callback.dataset)
    num_points_per_env = len(set_goal_callback.dataset)//env.num_envs
    for i in range(env.num_envs):
        dataset = set_goal_callback.dataset[i*num_points_per_env:(i+1)*num_points_per_env]
        env.env_method("set_dataset", dataset=dataset, indices=i)
    results_method = GenerateResults(env, set_goal_callback, save_video = False)
    results_method.run(set_goal_callback) #keep multiples of n_envs
