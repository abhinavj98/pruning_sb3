from tabnanny import verbose


from PPOAE.policies import ActorCriticWithAePolicy
from custom_callbacks import CustomEvalCallback, CustomTrainCallback
from PPOAE.ppo_ae import PPOAE
from gym_env_discrete import PruningEnv
from PPOAE.models import *

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import utils
import numpy as np
import cv2
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
# PARSE ARGUMENTS
import argparse
# from args import args_dict
from args_test import args_dict

# Start a wandb run with `sync_tensorboard=True`

# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)

import wandb
import os
import json
if os.path.exists("./keys.json"):
   with open("./keys.json") as f:
     os.environ["WANDB_API_KEY"] = json.load(f)["api_key"]
if not args.TESTING:
    wandb.init(
        # set the wandb project where this run will be logged
        project="test-ppo",
        sync_tensorboard = True,
        # track hyperparameters and run metadata
        config=args
    )

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def exp_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return (progress_remaining)**2 * initial_value

    return func

load_path = None
train_env_kwargs = {"renders" : args.RENDER, "tree_urdf_path" :  args.TREE_TRAIN_URDF_PATH, "tree_obj_path" :  args.TREE_TRAIN_OBJ_PATH, "action_dim" : args.ACTION_DIM_ACTOR,
                "maxSteps" : args.MAX_STEPS, "movement_reward_scale" : args.MOVEMENT_REWARD_SCALE, "action_scale" : args.ACTION_SCALE, "distance_reward_scale" : args.DISTANCE_REWARD_SCALE,
                "condition_reward_scale" : args.CONDITION_REWARD_SCALE, "terminate_reward_scale" : args.TERMINATE_REWARD_SCALE, "collision_reward_scale" : args.COLLISION_REWARD_SCALE, 
                "slack_reward_scale" : args.SLACK_REWARD_SCALE, "orientation_reward_scale" : args.ORIENTATION_REWARD_SCALE, "tree_count" : 1}

eval_env_kwargs =  {"renders" : False, "tree_urdf_path" :  args.TREE_TEST_URDF_PATH, "tree_obj_path" :  args.TREE_TEST_OBJ_PATH, "action_dim" : args.ACTION_DIM_ACTOR,
                "maxSteps" : args.EVAL_MAX_STEPS, "movement_reward_scale" : args.MOVEMENT_REWARD_SCALE, "action_scale" : args.ACTION_SCALE, "distance_reward_scale" : args.DISTANCE_REWARD_SCALE,
                "condition_reward_scale" : args.CONDITION_REWARD_SCALE, "terminate_reward_scale" : args.TERMINATE_REWARD_SCALE, "collision_reward_scale" : args.COLLISION_REWARD_SCALE, 
                "slack_reward_scale" : args.SLACK_REWARD_SCALE, "num_points" : args.EVAL_POINTS, "orientation_reward_scale" : args.ORIENTATION_REWARD_SCALE}

env = make_vec_env(PruningEnv, env_kwargs = train_env_kwargs, n_envs = 10)
new_logger = utils.configure_logger(verbose = 0, tensorboard_log = "./runs/", reset_num_timesteps = True)
env.logger = new_logger 
eval_env = Monitor(PruningEnv(**eval_env_kwargs))
eval_env.logger = new_logger
# Use deterministic actions for evaluation
eval_callback = CustomEvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=args.EVAL_FREQ,
                             deterministic=True, render=False,  n_eval_episodes = args.EVAL_EPISODES)
# It will check your custom environment and output additional warnings if needed
# check_env(env)

# video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
custom_callback = CustomTrainCallback()
policy_kwargs = {
        "actor_class":  Actor,
        "critic_class":  Critic,
        "actor_kwargs": {"state_dim": args.STATE_DIM, "emb_size": args.EMB_SIZE},
        "critic_kwargs": {"state_dim": args.STATE_DIM, "emb_size": args.EMB_SIZE},
        "features_extractor_class" : AutoEncoder,
        "optimizer_class" : th.optim.Adam,
	    "log_std_init" : args.LOG_STD_INIT,
        }

model = PPOAE(ActorCriticWithAePolicy, env, policy_kwargs=policy_kwargs, learning_rate=linear_schedule(args.LEARNING_RATE), learning_rate_ae=exp_schedule(args.LEARNING_RATE_AE), learning_rate_logstd = linear_schedule(0.01),\
              n_steps=args.STEPS_PER_EPOCH, batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS )
print(model.policy.parameters)
if load_path:
    model.load(load_path)
model.set_logger(new_logger)
print("Using device: ", utils.get_device())

env.reset()
model.learn(10000000, callback=[custom_callback, eval_callback], progress_bar = False)
