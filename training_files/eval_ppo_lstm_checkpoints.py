import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.pruning_gym.callbacks.eval_callbacks import CustomResultCallback
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.tree import Tree
# import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse
# from args import args_dict
from pruning_sb3.args.args import args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, make_or_bins, get_policy_kwargs
import random
if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)
    verbose = 1

    print(parsed_args_dict)
    load_timestep = args_global['load_timestep']
    or_bins = make_or_bins(args_test, "test", args_global['tree_type'])

    data_env_test = PruningEnv(**args_test, make_trees=True)
    or_bins_test = Tree.create_bins(18, 36)
    for key in or_bins_test.keys():
        for i in data_env_test.trees:
            or_bins_test[key].extend(i.or_bins[key])
    del data_env_test
    # Shuffle the data inside the bisn
    for key in or_bins_test.keys():
        random.shuffle(or_bins_test[key])

    eval_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=args_global["n_envs"])
    eval_callback = CustomResultCallback(eval_env, best_model_save_path="../logs/test",
                                         log_path="../logs/test",
                                         deterministic=True, render=False, or_bins=or_bins_test, save_video=False,
                                         **parsed_args_dict['args_callback'])
    mean_reward_list = []
    load_timestep_list = [2024000, 2112000, 2200000, 2288000]
    for i in range(len(load_timestep_list)):
        load_timestep = load_timestep_list[i]
        print("Loading model at timestep: ", load_timestep)
        if parsed_args_dict['args_global']['load_path']:
            load_path_model = "./logs/{}/model_{}_steps.zip".format(
                parsed_args_dict['args_global']['load_path'], load_timestep)
            # load_path_mean_std = "./logs/{}/current_mean_std_{}.pkl".format(
            #     parsed_args_dict['args_global']['load_path'], load_timestep)
        else:
            load_path_model = None

        policy_kwargs = get_policy_kwargs(args_policy, args_env, Encoder)
        policy = RecurrentActorCriticPolicy
        model = RecurrentPPOAE.load(load_path_model, env=eval_env)  # , custom_objects=load_dict)
        # model.policy.load_running_mean_std_from_file(load_path_mean_std)
        model.num_timesteps = load_timestep
        model._num_timesteps_at_start = load_timestep

        print(model.num_timesteps)
        print("Policy on device: ", model.policy.device)
        print("Model on device: ", model.device)
        print("Optical flow on device: ", model.policy.optical_flow_model.device)
        print("Using device: ", utils.get_device())
        eval_callback.model = model
        eval_callback._init_callback()
        mean_reward, std_reward = eval_callback.get_results()
        mean_reward_list.append((mean_reward, load_timestep))
        print("Mean reward: ", mean_reward)

    print(mean_reward_list)
