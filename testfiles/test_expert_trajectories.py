import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import random
import argparse
from pruning_sb3.args.args import args
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
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)
    verbose = args_callback['verbose']

    load_timestep = args_global['load_timestep']

    env = PruningEnv(**args_record)
    expert_trajectory_path = "trajectories_test.hdf5"
    with h5py.File(expert_trajectory_path, 'r') as file:
        traj_names = list(file.keys())
    #Loop through the all the datasets in the hdf5 file
    print("Trajectories: ", len(traj_names))

    for dname in traj_names: #{"trajectory_0", "trajectory_1", "trajectory_2"
        with h5py.File(expert_trajectory_path, 'r') as file:
            observation_dict = {}
            expert_traj = file[dname]
            # tree_info = expert_traj['tree_info']
            #Print attributes of the dataset
            print("Expert trajectory: ", dname)
            print(expert_traj.attrs.keys())
            for attr_name, attr_value in expert_traj.attrs.items():
                print(f"Attribute: {attr_name} -> {attr_value} -> {type(attr_value)}")
            env.set_tree_properties(tree_urdf=expert_traj.attrs['tree_urdf'],point_pos=expert_traj.attrs['point_pos'],
                                    point_branch_or=expert_traj.attrs['point_branch_or'],tree_orientation=expert_traj.attrs['tree_orientation'],
                                    tree_scale=expert_traj.attrs['tree_scale'],tree_pos=expert_traj.attrs['tree_pos'],
                                    point_branch_normal=expert_traj.attrs['point_branch_normal'])
            env.set_ur5_pose(expert_traj.attrs['robot_pos'], expert_traj.attrs['robot_or'])
            actions = expert_traj['actions']
            #read all actions from the dataset
            actions = actions[:]
            observations = expert_traj['observations']
            for key, value in observations.items():
                observation_dict[key] = value[:]

            # next_observations = expert_traj['next_observations']
            rewards = expert_traj['rewards']
            rewards = rewards[:]
            dones = expert_traj['dones']
            dones = dones[:]
            # print("Tree info: ", tree_info)
            print("Actions: ", actions.shape)
            print("Observations: ", observations)
            print("Rewards: ", rewards.shape)
            print("Dones: ", dones.shape)
            # print("Next observations: ", next_observations)

        for i in range(len(actions)):
            action = actions[i]
            print("Action: ", action)
            observation_dict = {}

            # env.set_observation(observation)
            # env.set_action(action)
            obs, rew, term, trunc, _ = env.step(action)
            print("env", rew, term)
            print("file", rewards[i], dones[i])

    # Get dataset names
     #shuffle the expert trajectories
    # # random.shuffle(expert_trajectories)
    # for expert_trajectory in expert_trajectories:
    #     print("Expert trajectory: ", expert_trajectory)
    #     with open(expert_trajectory, "rb") as f:
    #         expert_data = pickle.load(f)
    #     print("Expert data: ", expert_data['actions'])
    #     tree_info = expert_data['tree_info']
    #     actions = expert_data['actions']
    #     observations = expert_data['observations']
    #     rewards = expert_data['rewards']
    #     dones = expert_data['dones']
    #     env.set_tree_properties(*tree_info)
    #
    #     # env.ur5.reset_ur5_arm()
    #     env.reset()
    #     for i in range(len(actions)):
    #         action = actions[i]
    #         print("Action: ", action)
    #         observation = observations[i]
    #         # env.set_observation(observation)
    #         # env.set_action(action)
    #         obs, rew, term, trunc, _ = env.step(action)
    #         print("env", rew, term)
    #         print("file", rewards[i], dones[i])
    #     input()
    #
    # print("Env created")
