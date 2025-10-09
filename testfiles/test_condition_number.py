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
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    or_bins = make_or_bins(args_test, "test", args_global['tree_type'])

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
    set_goal_callback._update_tree_properties()
    env.reset()

    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:
        # tf = env.ur5.get_camera_location(env.cam_pan, env.cam_tilt, env.cam_xyz_offset)
        # # orientation = np.array(env.pyb.con.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # orientation = tf[:3, :3]
        # loc = tf[:3, 3]
        # print(env.ur5.get_view_mat_at_curr_pose(0,0,0))
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('j') in action:  # +X
            val = np.array([0.05, 0, 0, 0, 0, 0])
        elif ord('l') in action:  # -X
            val = np.array([-0.05, 0, 0, 0, 0, 0])
        elif ord('k') in action:  # +Y
            val = np.array([0, 0.05, 0, 0, 0, 0])
        elif ord('i') in action:  # -Y
            val = np.array([0, -0.05, 0, 0, 0, 0])
        elif ord('u') in action:  # +Z
            val = np.array([0, 0, 0.05, 0, 0, 0])
        elif ord('o') in action:  # -Z
            val = np.array([0, 0, -0.05, 0, 0, 0])
        elif ord('n') in action:  # +Roll (Rx)
            val = np.array([0, 0, 0, 0.05, 0, 0])
        elif ord('m') in action:  # -Roll (Rx)
            val = np.array([0, 0, 0, -0.05, 0, 0])
        elif ord('t') in action:  # +Pitch (Ry)
            val = np.array([0, 0, 0, 0, 0.05, 0])
        elif ord('y') in action:  # -Pitch (Ry)
            val = np.array([0, 0, 0, 0, -0.05, 0])
        elif ord('h') in action:  # +Yaw (Rz)
            val = np.array([0, 0, 0, 0, 0, 0.05])
        elif ord('b') in action:  # -Yaw (Rz)
            # env.force_time_limit()
            infos = {}
            infos['TimeLimit.truncated'] = True
            set_goal_callback.locals = {"infos": [infos]}
            set_goal_callback._update_tree_properties()
            env.reset()
            # env.is_goal_state = True
        else:
            val = np.array([0.,0.,0, 0., 0., 0.])



        val_loc = env.convert_global_action_to_local(val)
        # val =  np.hstack((global_velocity, global_angular_velocity))
        # print()
        observation, reward, terminated, truncated, infos = env.step(val_loc)
        # print("infos", infos)
        set_goal_callback.locals = {"infos": [infos]}
        # print(np.array(env.pyb.con.getMatrixFromQuaternion(orientation)).reshape(3, 3))
        # trans, ang = env.ur5.get_current_vel(env.ur5.end_effector_index)
        # print("Current velocity in ee frame", np.dot(orientation.T,trans))
        # print("Current angular velocity in ee frame", np.dot(orientation.T,ang))
        # env.pyb.visualize_rot_mat(orientation, loc)
        # print("Current pose", env.ur5.get_current_pose(env.ur5.end_effector_index))
        time.sleep(0.05)
        # print(env.ur5.check_)
        # print(env.ur5.get_joint_angles())
        ee_list = [env.ur5.tool0_index]

        for i in ee_list:
            pos, orn = env.ur5.get_current_pose(i)
            orn_mat = np.array(env.pyb.con.getMatrixFromQuaternion(orn)).reshape(3, 3)
            env.pyb.visualize_rot_mat(orn_mat, pos)
            # print(env.ur5.calculate_jacobian())
        # input()
        # # print("Velocity global", val, val_loc, env.ur5.get_current_vel(env.ur5.tool0_index))
        # # print("Current pose", env.ur5.get_current_pose(env.ur5.end_effector_index))
        # # print(infos)
        # input()
        # pos, orn = env.ur5.get_current_pose(env.ur5.end_effector_index)
        # pos = list(pos)
        # pos[1] = pos[1] + 0.025
        # # # camera_tf = env.ur5.create_camera_transform(0, np.pi/180*10, np.array([0,0,0]))
        # # orn_mat = np.array(env.pyb.con.getMatrixFromQuaternion(orn)).reshape(3, 3)
        # # # print(camera_tf)
        # env.pyb.visualize_rot_mat(orn_mat, pos)
        #
        # pos, orn = env.ur5.get_current_pose(3)
        # # camera_tf = env.ur5.create_camera_transform(0, np.pi/180*10, np.array([0,0,0]))
        # orn_mat = np.array(env.pyb.con.getMatrixFromQuaternion(orn)).reshape(3, 3)
        # # print(camera_tf)
        # env.pyb.visualize_rot_mat(orn_mat, pos)
        # camera_tf = env.ur5.create_camera_transform(env.cam_pan, env.cam_tilt, env.cam_xyz_offset)
        # env.pyb.visualize_rot_mat(camera_tf[:3, :3], camera_tf[:3, 3])
        # condition_number = env.ur5.get_condition_number()
        # print("Condition number", condition_number)
        # curr_pose = env.ur5.get_current_pose(env.ur5.end_effector_index)
        # print("Current pose", curr_pose)