from tabnanny import verbose

from a2c import A2CWithAE
from a2c_with_ae_policy import ActorCriticWithAePolicy
from custom_callbacks import CustomEvalCallback, CustomCallback
from ppo_ae import PPOAE
from gym_env_discrete import ur5GymEnv
from models import *

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import utils
import numpy as np
import cv2
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# PARSE ARGUMENTS
import argparse
from args import args_dict


# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)

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
# set up logger


        # Create eval callback if needed
n_envs = 8
load_path = None
env_kwargs = {"renders" : args.RENDER, "tree_urdf_path" :  args.TREE_TRAIN_URDF_PATH, "tree_obj_path" :  args.TREE_TRAIN_OBJ_PATH, "action_dim" : args.ACTION_DIM_ACTOR}
env = make_vec_env(ur5GymEnv, env_kwargs = env_kwargs, n_envs = args.N_ENVS)
new_logger = utils.configure_logger(verbose = 0, tensorboard_log = "./runs/", reset_num_timesteps = True)
env.logger = new_logger 
eval_env = ur5GymEnv(renders=False, tree_urdf_path= args.TREE_TEST_URDF_PATH, tree_obj_path=args.TREE_TEST_OBJ_PATH, name = "evalenv", num_points = args.EVAL_POINTS)

# Use deterministic actions for evaluation
eval_callback = CustomEvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=args.EVAL_FREQ,
                             deterministic=True, render=False,  n_eval_episodes = args.EVAL_EPISODES)
# It will check your custom environment and output additional warnings if needed
# check_env(env)

# video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
custom_callback = CustomCallback()
policy_kwargs = {
        "actor_class":  Actor,
        "critic_class":  Critic,
        "actor_kwargs": {"state_dim": args.STATE_DIM, "emb_size": args.EMB_SIZE},
        "critic_kwargs": {"state_dim": args.STATE_DIM, "emb_size": args.EMB_SIZE},
        "features_extractor_class" : AutoEncoder,
        "optimizer_class" : th.optim.Adam
        }

model = PPOAE(ActorCriticWithAePolicy, env, policy_kwargs=policy_kwargs, learning_rate=linear_schedule(args.LEARNING_RATE), learning_rate_ae=exp_schedule(args.LEARNING_RATE),\
              n_steps=args.STEPS_PER_EPOCH, batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS, )
if load_path:
    model.load(load_path)
model.set_logger(new_logger)
print("Using device: ", utils.get_device())

env.reset()
for _ in range(100):
    #env.render() 
    env.step([env.action_space.sample()]*n_envs) # take a random action
model.learn(10000000, callback=[custom_callback, eval_callback], progress_bar = False)
