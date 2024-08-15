import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import random
import argparse
from pruning_sb3.args.args_test import args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, set_args, organize_args
from pruning_sb3.pruning_gym.helpers import make_or_bins, get_policy_kwargs
from pruning_sb3.pruning_gym.callbacks.train_callbacks import PruningTrainSetGoalCallback
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from stable_baselines3.common import utils
from pruning_sb3.pruning_gym.tree import Tree
import time
import pickle
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)
    verbose = args_callback['verbose']

    load_timestep = args_global['load_timestep']

    env = PruningEnv(**args_record)

    expert_trajectory_path = "expert_trajectories"
    expert_trajectories = glob.glob(expert_trajectory_path + "/*.pkl")
    #shuffle the expert trajectories
    random.shuffle(expert_trajectories)
    for expert_trajectory in expert_trajectories:
        print("Expert trajectory: ", expert_trajectory)
        with open(expert_trajectory, "rb") as f:
            expert_data = pickle.load(f)
        print("Expert data: ", expert_data['actions'])
        tree_info = expert_data['tree_info']
        actions = expert_data['actions']
        observations = expert_data['observations']
        # dones = expert_data['dones']
        env.set_tree_properties(*tree_info)

        # env.ur5.reset_ur5_arm()
        env.reset()
        for i in range(len(actions)):
            action = actions[i]*10
            observation = observations[i]
            # env.set_observation(observation)
            # env.set_action(action)
            obs, rew, term, trunc, _ = env.step(action)
            print(rew, term)
        input()

    print("Env created")
