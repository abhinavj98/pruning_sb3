import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.pruning_gym.custom_callbacks import CustomResultCallback
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import AutoEncoder
from pruning_sb3.pruning_gym.tree import Tree
# import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse
# from args import args_dict
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
import random

if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    parsed_args_dict = organize_args(parsed_args)
    shared_tree_list_train = []
    shared_tree_list_test = []

    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/best_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "./logs/{}/best_mean_std.pkl".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path_model = None

    # add_arg_to_env('shared_tree_list', shared_list, ['args_train'], parsed_args_dict)
    # Duplicates are resolved in favor of the value in x; dict(y, **x)
    args_global = parsed_args_dict['args_global']
    args_train = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_train'])
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    args_record = dict(args_test, **parsed_args_dict['args_record'])


    data_env_test = PruningEnv(**args_test, make_trees=True)
    or_bins_test = Tree.create_bins(18, 36)
    for key in or_bins_test.keys():
        for i in data_env_test.trees:
            or_bins_test[key].extend(i.or_bins[key])
    del data_env_test
    #Shuffle the data inside the bisn
    for key in or_bins_test.keys():
        random.shuffle(or_bins_test[key])
    args_train["n_eval_episodes"] = len(or_bins_test.keys())
    eval_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=8)
    # Use deterministic actions for evaluation
    eval_callback = CustomResultCallback(eval_env,  best_model_save_path="../logs/test",
                                       log_path="../logs/test",
                                       deterministic=True, render=False, or_bins = or_bins_test, **parsed_args_dict['args_callback'])

    policy_kwargs = {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": parsed_args_dict['args_policy']['state_dim'],
                                      "in_channels": 3, },
        "optimizer_class": th.optim.Adam,
        "log_std_init": parsed_args_dict['args_policy']['log_std_init'],
        "net_arch": dict(
            qf=[parsed_args_dict['args_policy']['emb_size'] * 2, parsed_args_dict['args_policy']['emb_size'],
                parsed_args_dict['args_policy']['emb_size'] // 2],
            pi=[parsed_args_dict['args_policy']['emb_size'] * 2, parsed_args_dict['args_policy']['emb_size'],
                parsed_args_dict['args_policy']['emb_size'] // 2]),
        "share_features_extractor": True,
        "n_lstm_layers": 2,
    }
    policy = RecurrentActorCriticPolicy
    model = RecurrentPPOAE.load(load_path_model, env=eval_env,  device=th.device('cpu'))#, custom_objects=load_dict)
    model.policy.load_running_mean_std_from_file(load_path_mean_std)
    model.num_timesteps = 100000
    model._num_timesteps_at_start = 100000

    print(model.num_timesteps)
    print("Policy on device: ", model.policy.device)
    print("Model on device: ", model.device)
    print("Optical flow on device: ", model.policy.optical_flow_model.device)
    print("Using device: ", utils.get_device())
    eval_callback.model = model
    eval_callback._init_callback()
    eval_callback.get_results()

