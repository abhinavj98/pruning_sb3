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

def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.pyb.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)
    verbose = args_callback['verbose']

    load_timestep = args_global['load_timestep']

    if args_global['load_path']:
        load_path_model = "./logs/{}/model_{}_steps.zip".format(
            args_global['load_path'], load_timestep)
        load_path_mean_std = "./logs/{}/model_mean_std_{}_steps.pkl".format(
            args_global['load_path'], load_timestep)
    else:
        load_path_model = None

    print(parsed_args_dict)
    or_bins = make_or_bins(args_test, "test")

    env = PruningEnv(**args_record)
    print("Env created")
    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy

    model = RecurrentPPOAE(policy, env, policy_kwargs=policy_kwargs,
                           learning_rate=linear_schedule(parsed_args_dict['args_policy']['learning_rate']),
                           learning_rate_ae=exp_schedule(parsed_args_dict['args_policy']['learning_rate_ae']),
                           learning_rate_logstd=None,
                           n_steps=parsed_args_dict['args_policy']['steps_per_epoch'],
                           batch_size=parsed_args_dict['args_policy']['batch_size'],
                           n_epochs=parsed_args_dict['args_policy']['epochs'])
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger
    model.set_logger(new_logger)
    set_goal_callback = PruningTrainSetGoalCallback(or_bins=or_bins, verbose=args_callback['verbose'])
    set_goal_callback.init_callback(model)
    env.action_scale = 1
    # env.ur5.set_joint_angles((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485, -0.7777109569760938, 3.210501267258541))
    infos = [{}]
    infos[0]['TimeLimit.truncated'] = True

    set_goal_callback.update_locals(locals())
    env.reset()
    set_goal_callback._update_tree_properties()

    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:

        loc, orientation = env.robot.get_current_pose(env.robot.end_effector_index)
        orientation = np.array(env.pyb.con.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('a') in action:
            val = np.array([0.1, 0, 0, 0, 0, 0])
        elif ord('d') in action:
            val = np.array([-0.1, 0, 0, 0, 0, 0])
        elif ord('s') in action:
            val = np.array([0, 0.1, 0, 0, 0, 0])
        elif ord('w') in action:
            val = np.array([0, -0.1, 0, 0, 0, 0])
        elif ord('q') in action:
            val = np.array([0, 0, 0.1, 0, 0, 0])
        elif ord('e') in action:
            val = np.array([0, 0, -0.1, 0, 0, 0])
        elif ord('z') in action:
            val = np.array([0, 0, 0, 0.1, 0, 0])
        elif ord('c') in action:
            val = np.array([0, 0, 0, -0.1, 0, 0])
        elif ord('x') in action:
            val = np.array([0, 0, 0, 0, 0.1, 0])
        elif ord('v') in action:
            val = np.array([0, 0, 0, 0, -0.1, 0])
        elif ord('r') in action:
            val = np.array([0, 0, 0, 0, 0, 0.5])
        elif ord('f') in action:
            val = np.array([0, 0, 0, 0, 0, -0.5])
        elif ord('t') in action:
            # env.force_time_limit()
            infos = {}
            infos['TimeLimit.truncated'] = True
            env.reset()
            set_goal_callback._update_tree_properties()
            # env.is_goal_state = True
        else:
            val = np.array([0.,0.,0, 0., 0., 0.])

        # global_velocity = np.dot(orientation, val[:3])
        # global_angular_velocity = np.dot(orientation, val[3:])
        #
        # val =  np.hstack((global_velocity, global_angular_velocity))

        observation, reward, terminated, truncated, infos = env.step(val)
        set_goal_callback.locals = {"infos": [infos]}
        print(env.robot.get_joint_angles())
        # print(np.array(env.pyb.con.getMatrixFromQuaternion(orientation)).reshape(3, 3))
        # trans, ang = env.robot.get_current_vel(env.robot.end_effector_index)
        # print("Current velocity in ee frame", np.dot(orientation.T,trans))
        # print("Current angular velocity in ee frame", np.dot(orientation.T,ang))
        env.pyb.visualize_rot_mat(orientation, loc)
        time.sleep(0.1)

