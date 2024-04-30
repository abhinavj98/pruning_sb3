import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import cv2
import random
import argparse
from pruning_sb3.args.args_test import args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
import multiprocessing as mp
import copy
from pruning_sb3.pruning_gym.custom_callbacks import CustomTrainCallback
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from stable_baselines3.common import utils
from pruning_sb3.pruning_gym.tree import Tree
def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.pyb.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)
if __name__ == "__main__":
    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/current_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "./logs/{}/current_mean_std.pkl".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path_model = None
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    args_test['renders'] = False
    or_bins_test = Tree.create_bins(18, 36)
    data_env_test = PruningEnv(**args_test, make_trees=True)
    for key in or_bins_test.keys():
        for i in data_env_test.trees:
            or_bins_test[key].extend(i.or_bins[key])
    del data_env_test
    # Shuffle the data inside the bisn
    for key in or_bins_test.keys():
        random.shuffle(or_bins_test[key])
    args_test['renders'] = True
    args_test['max_steps'] = 100000
    env = PruningEnv(**args_test)

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
    train_callback = CustomTrainCallback(or_bins = or_bins_test)
    train_callback.init_callback(model)
    train_callback.on_training_start(locals(), globals())

    env.action_scale = 1
    # env.ur5.set_joint_angles((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485, -0.7777109569760938, 3.210501267258541))
    env.reset()
    for _ in range(100):
        env.pyb.con.stepSimulation()
    # env.reset()
    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('a') in action:
            val = np.array([0.01, 0, 0, 0, 0, 0])
        elif ord('d') in action:
            val = np.array([-0.01, 0, 0, 0, 0, 0])
        elif ord('s') in action:
            val = np.array([0, 0.01, 0, 0, 0, 0])
        elif ord('w') in action:
            val = np.array([0, -0.01, 0, 0, 0, 0])
        elif ord('q') in action:
            val = np.array([0, 0, 0.01, 0, 0, 0])
        elif ord('e') in action:
            val = np.array([0, 0, -0.01, 0, 0, 0])
        elif ord('z') in action:
            val = np.array([0, 0, 0, 0.01, 0, 0])
        elif ord('c') in action:
            val = np.array([0, 0, 0, -0.01, 0, 0])
        elif ord('x') in action:
            val = np.array([0, 0, 0, 0, 0.01, 0])
        elif ord('v') in action:
            val = np.array([0, 0, 0, 0, -0.01, 0])
        elif ord('r') in action:
            val = np.array([0, 0, 0, 0, 0, 0.05])
        elif ord('f') in action:
            val = np.array([0, 0, 0, 0, 0, -0.05])
        elif ord('t') in action:
            env.reset()
            env.force_time_limit()
        else:
            val = np.array([0.,0.,0., 0., 0., 0.])
        # print(val)
        observation, reward, terminated, truncated, infos = env.step(val)

        # base_pos, base_quat = p.getBasePositionAndOrientation(robot)
        #get base position and orientation

        # print("ee position", env.ur5.get_current_pose(env.ur5.end_effector_index)[0])
        # print("tool 0 position", env.ur5.get_current_pose(env.ur5.tool0_link_index)[0])
        # print("tree position", env.tree.pos)
        # print("Base position", env.ur5.get_current_pose(env.ur5.base_index)[0])
        #pring joint angles and condition number
        # print(env.ur5.get_joint_angles())
        # print(env.ur5.get_condition_number())
        base_pos, base_quat = env.pyb.con.getBasePositionAndOrientation(env.ur5.ur5_robot)
        print(infos['TimeLimit.truncated'])
        infos = [infos]

        train_callback.on_step()
        # print(base_pos, base_quat)
        # print(env.ur5.get_current_pose(0))
        # print(env.ur5.get_joint_angles())
        # print(env.con.getLinkState(env.ur5, env.end_effector_index, 1)[6])
        # print(env.con.getLinkState(env.ur5, env.end_effector_index, 1)[7])

        # print(env.get_current_pose())
        # print(infos)
        # print(observation['desired_goal'], observation['achieved_goal'])
        # env.render()
        # jacobian = env.pyb_con.con.calculateJacobian(env.ur5.ur5_robot, env.ur5.end_effector_index, [0, 0, 0],
        #                                      env.ur5.get_joint_angles(), [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        # jacobian = np.vstack(jacobian)
        # condition_number = np.linalg.cond(jacobian)
        # print("as", jacobian)
        # jacobian = env.pyb_con.con.calculateJacobian(env.ur5.ur5_robot, env.ur5.tool0_link_index, [0, 0, 0],
        #                                      env.ur5.get_joint_angles(), [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        # jacobian = np.vstack(jacobian)
        # condition_number = np.linalg.cond(jacobian)
        # print(jacobian)
        # print(condition_number, 1 / condition_number)
        # # print(env.get_joint_angles())
        # print(env.target_dist)

    """
    print("Initial position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
    try:
        action = int(input('action please'))
    except:
        continue
    if action == 0:
        quit()
    if action > 12:
        print("Wrong action")
        continue
    print(env.rev_actions[action])
    
    r = env.step(action, False)
    print(r[1][-1])
    print("Final position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
    """
