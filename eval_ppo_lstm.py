# Description: Evaluate the trained model
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CallbackList
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import cv2
import torch as th
from gym_env_discrete import PruningEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor
from PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
import argparse
from args import args_dict
from helpers import optical_flow_create_shared_vars
import pandas as pd
class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        model : RecurrentPPOAE,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):

        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self._reward_dict = {}
        self.model = model
        self.n_eval_episodes = 1
    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _log_rewards_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]
        #add to buffers
        self._reward_dict["movement_reward"].append(infos["movement_reward"])
        self._reward_dict["distance_reward"].append(infos["distance_reward"])
        self._reward_dict["terminate_reward"].append(infos["terminate_reward"])
        self._reward_dict["collision_reward"].append(infos["collision_reward"])
        self._reward_dict["slack_reward"].append(infos["slack_reward"])
        self._reward_dict["condition_number_reward"].append(infos["condition_number_reward"])
        self._reward_dict["velocity_reward"].append(infos["velocity_reward"])
        self._reward_dict["pointing_orientation_reward"].append(infos["pointing_orientation_reward"])
        self._reward_dict["perpendicular_orientation_reward"].append(infos["perpendicular_orientation_reward"])

    def _log_final_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        infos = locals_["info"]
        if infos["TimeLimit.truncated"] or infos["is_success"]:
            self._reward_dict["pointing_cosine_sim_error"].append(infos["pointing_cosine_sim_error"])
            self._reward_dict["perpendicular_cosine_sim_error"].append(infos["perpendicular_cosine_sim_error"])
            self._reward_dict["euclidean_error"].append(infos["euclidean_error"])

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._collisions_buffer.append(self.eval_env.get_attr("collisions", 0)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)
        self._log_final_metrics(_locals, _globals)

    def eval_policy(self):

        # Reset success rate buffer
        self._is_success_buffer = []
        self._screens_buffer = []
        self._collisions_buffer = []
        self._reward_dict["movement_reward"] = []
        self._reward_dict["distance_reward"] = []
        self._reward_dict["terminate_reward"] = []
        self._reward_dict["collision_reward"] = []
        self._reward_dict["slack_reward"] = []
        self._reward_dict["condition_number_reward"] = []
        self._reward_dict["velocity_reward"] = []
        self._reward_dict["orientation_reward"] = []
        self._reward_dict["pointing_orientation_reward"] = []
        self._reward_dict["perpendicular_orientation_reward"] = []
        self._reward_dict["pointing_cosine_sim_error"] = []
        self._reward_dict["perpendicular_cosine_sim_error"] = []
        self._reward_dict["euclidean_error"] = []

        print(self.render)
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._master_callback,
        )

        reward_df = pd.DataFrame(self._reward_dict)
        success_df = pd.DataFrame(self._is_success_buffer)
        reward_df.append(success_df)
        reward_df.to_csv('out.csv', index=False)
        print("Success", np.mean(self._is_success_buffer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments to the parser based on the dictionary
    for arg_name, arg_params in args_dict.items():
        parser.add_argument(f'--{arg_name}', **arg_params)

    # Parse arguments from the command line
    args = parser.parse_args()
    print(args)

    optical_flow_subproc = True
    if args.USE_OPTICAL_FLOW and optical_flow_subproc:
        print("Using optical flow")
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)

    if args.LOAD_PATH:
        print("Loading model from {}".format(args.LOAD_PATH))
        load_path =  "./logs/best_model"#./logs/{}/best_model.zip".format(
            #args.LOAD_PATH)  # ./nfs/stak/users/jainab/hpc-share/codes/pruning_sb3/logs/lowlr/best_model.zip"#Nonei
    else:
        load_path = None




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
                       "name": "evalenv", "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                       "shared_var": shared_var}
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(device)
    eval_env = Monitor(PruningEnv(**eval_env_kwargs))
    model = RecurrentPPOAE.load(load_path)
    evaluate_policy(model, eval_env, n_eval_episodes=1, render=False, deterministic=True)
    # eval = CustomEvalCallback(eval_env, model)
    # eval.eval_policy()