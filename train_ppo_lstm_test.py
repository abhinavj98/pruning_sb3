from typing import Callable, Union

from PPOLSTMAE.policies import RecurrentActorCriticPolicy
from custom_callbacks import CustomEvalCallback, CustomTrainCallback
from PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from gym_env_discrete import PruningEnv
from models import AutoEncoder
# import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse
# from args import args_dict
from args_test import args_dict
import random
import multiprocessing as mp
from optical_flow import OpticalFlow

from helpers import init_wandb, linear_schedule, exp_schedule, optical_flow_create_shared_vars

# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    #TODO: put in args
    optical_flow_subproc = True
    if args.USE_OPTICAL_FLOW and optical_flow_subproc:
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)

    if args.LOAD_PATH:
        load_path = "logs/run/best_model.zip"  # "./logs/{}/best_model.zip".format(args.LOAD_PATH)#./nfs/stak/users/jainab/hpc-share/codes/pruning_sb3/logs/lowlr/best_model.zip"#Nonei
    else:
        load_path = None
    train_env_kwargs = {"renders": args.RENDER, "tree_urdf_path": args.TREE_TRAIN_URDF_PATH,
                        "tree_obj_path": args.TREE_TRAIN_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                        "max_steps": args.MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                        "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                        "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                        "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                        "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                        "slack_reward_scale": args.SLACK_REWARD_SCALE,
                        "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                        "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                        "tree_count": 1, "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                        "shared_var": shared_var}

    eval_env_kwargs = {"renders": False, "tree_urdf_path": args.TREE_TEST_URDF_PATH,
                       "tree_obj_path": args.TREE_TEST_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                       "max_steps": args.EVAL_MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                       "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                       "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                       "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                       "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                       "slack_reward_scale": args.SLACK_REWARD_SCALE, "num_points": args.EVAL_POINTS,
                       "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                       "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                       "name": "evalenv", "optical_flow_subproc": True, "use_optical_flow": args.USE_OPTICAL_FLOW,
                       "shared_var": shared_var}

    record_env_kwargs = {"renders": False, "tree_urdf_path": args.TREE_TEST_URDF_PATH,
                         "tree_obj_path": args.TREE_TEST_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                         "max_steps": args.EVAL_MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                         "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                         "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                         "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                         "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                         "slack_reward_scale": args.SLACK_REWARD_SCALE, "num_points": args.EVAL_POINTS,
                         "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                         "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                         "name": "recordenv", "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                         "shared_var": shared_var}

    env = make_vec_env(PruningEnv, env_kwargs=train_env_kwargs, n_envs=args.N_ENVS, vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger
    eval_env = make_vec_env(PruningEnv, env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, n_envs=args.N_ENVS)
    record_env = make_vec_env(PruningEnv, env_kwargs=record_env_kwargs, vec_env_cls=SubprocVecEnv, n_envs=1)
    eval_env.logger = new_logger
    # Use deterministic actions for evaluation
    eval_callback = CustomEvalCallback(eval_env, record_env, best_model_save_path="./logs/test",
                                       log_path="./logs/test", eval_freq=args.EVAL_FREQ,
                                       deterministic=True, render=False, n_eval_episodes=args.EVAL_EPISODES)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    # video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
    train_callback = CustomTrainCallback()

    policy_kwargs = {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": args.STATE_DIM,
                                      "in_channels": (3 if args.USE_OPTICAL_FLOW else 1), },
        "optimizer_class": th.optim.Adam,
        "log_std_init": args.LOG_STD_INIT,
        "net_arch": dict(qf=[args.EMB_SIZE * 2, args.EMB_SIZE, args.EMB_SIZE // 2],
                         pi=[args.EMB_SIZE * 2, args.EMB_SIZE, args.EMB_SIZE // 2]),
        "share_features_extractor": True,
        "n_lstm_layers": 2,
    }
    policy = RecurrentActorCriticPolicy

    # model = RecurrentPPOAE(policy, env, policy_kwargs = policy_kwargs, learning_rate = linear_schedule(args.LEARNING_RATE), learning_rate_ae=exp_schedule(args.LEARNING_RATE_AE), learning_rate_logstd = linear_schedule(0.01), n_steps=args.STEPS_PER_EPOCH, batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS)
    if load_path:
        model = RecurrentPPOAE.load(load_path, env)
        model.num_timesteps = 100000
        model._num_timesteps_at_start = 100000
        print(model.num_timesteps)
    else:
        model = RecurrentPPOAE(policy, env, policy_kwargs=policy_kwargs,
                               learning_rate=linear_schedule(args.LEARNING_RATE),
                               learning_rate_ae=exp_schedule(args.LEARNING_RATE_AE),
                               learning_rate_logstd=linear_schedule(0.01), n_steps=args.STEPS_PER_EPOCH,
                               batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS)

    model.set_logger(new_logger)
    print("Using device: ", utils.get_device())

    model.learn(10000000, callback=[train_callback, eval_callback], progress_bar=False, reset_num_timesteps=False)
