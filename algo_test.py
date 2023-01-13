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
render = False
env = ur5GymEnv(renders=render)

new_logger = utils.configure_logger(verbose = 0, tensorboard_log = "./runs/", reset_num_timesteps = True)
env.logger = new_logger 
eval_env = ur5GymEnv(renders=False, name = "evalenv")

# Use deterministic actions for evaluation
eval_callback = CustomEvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False,  n_eval_episodes = 50)
# It will check your custom environment and output additional warnings if needed
# check_env(env)

# video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
custom_callback = CustomCallback()
policy_kwargs = {
        "actor_class":  Actor,
        "critic_class":  Critic,
        "actor_kwargs": {"state_dim": 72+10*3, "emb_size":128, "action_dim":10, "action_std":1},
        "critic_kwargs": {"state_dim": 72+10*3, "emb_size":128, "action_dim":1, "action_std":1},
        "features_extractor_class" : Encoder,
        "optimizer_class" : th.optim.Adam
        }

model = PPOAE(ActorCriticWithAePolicy, env, policy_kwargs=policy_kwargs, learning_rate=linear_schedule(0.001), learning_rate_ae=exp_schedule(0.001))
model.set_logger(new_logger)
print("Using device: ", utils.get_device())

env.reset()
for _ in range(100):
    env.render() 
    env.step(env.action_space.sample()) # take a random action
model.learn(1000000, callback=[custom_callback, eval_callback], progress_bar = False)
