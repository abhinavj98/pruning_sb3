from tabnanny import verbose
import gym
from a2c_with_ae_algo import A2CWithAE
from a2c_with_ae_policy import ActorCriticWithAePolicy
from gym_env_discrete import ur5GymEnv
from models import *
from typing import Any, Dict
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import cv2
from stable_baselines3.common.logger import configure

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


model_path = f"./logs/best_model.zip"
env = ur5GymEnv(renders=True)
# eval_env = ur5GymEnv(renders=False, eval=True)
new_logger = utils.configure_logger(verbose = 0, tensorboard_log = "./runs/", reset_num_timesteps = True)
env.logger = new_logger 

policy_kwargs = {
        "actor_model":  Actor(None, 10*3, 32, 12, 1),
        "critic_model":  Critic(None, 10*3, 32,1, 1),
        "features_extractor_class" : AutoEncoder,
        "optimizer_class" : th.optim.Adam
        }#ActorCriticWithAePolicy(env.observation_space, env.action_space, linear_schedule(0.001), Actor(None, 128*7*7+10*3,128, 12, 1 ), Critic(None, 128*7*7+10*3, 128,1,1), features_extractor_class =  AutoEncoder)
model = A2CWithAE(ActorCriticWithAePolicy, env, policy_kwargs=policy_kwargs, learning_rate=1e-3)
# model = model.load(model_path, env=env)
model.set_logger(new_logger)
# model.policy.load()
print("Using device: ", utils.get_device())
mean_reward, std_reward = evaluate_policy(
                model,
                env,
                5,
                True,
            )