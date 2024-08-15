import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pruning_sb3.pruning_gym.pruning_env import PruningEnvRRT
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, make_or_bins, convert_string
import argparse
import pandas as pd
parser = argparse.ArgumentParser()



if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)

    print(parsed_args_dict)
    or_bins = make_or_bins(args_train, "train")
    path_file = args_baseline['load_file_path']+'.csv'
    env = make_vec_env(PruningEnvRRT, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)

    paths_df = pd.read_csv(path_file)
    paths_success = paths_df[paths_df['is_success'] == True]
    paths_success['tree_info'] = paths_success['tree_info'].apply(convert_string)
    paths_success['path'] = paths_success['path'].apply(convert_string)
    # paths_success = paths_success[624:]
    num_paths = len(paths_success)
    num_points_per_env = len(paths_success)//env.num_envs

    for i in range(env.num_envs):
        dataset = paths_success.iloc[i*num_points_per_env:(i+1)*num_points_per_env]
        env.env_method("set_dataset", dataset=dataset, indices=i)


    controllable_joints = [3, 4, 5, 6, 7, 8]
    env.env_method("run_smoothing", save_video = args_baseline['save_video'], save_path = None)