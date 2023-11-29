# Description: Evaluate the trained model
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from stable_baselines3.common.callbacks import EventCallback
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from typing import Any, Dict, Union
import numpy as np
import torch as th
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
import argparse
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env, init_wandb

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
        n_eval_episodes: int = 10,
    ):
        np.random.seed(0)
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
        self._is_success_buffer = {}
        self.evaluations_successes = []
        self._reward_dict = {}
        self._reward_dict_temp = {}
        self._info_dict = {}
        self.model = model
        self.n_eval_episodes = n_eval_episodes
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
                self._is_success_buffer["is_success"].append(maybe_is_success)

    def _log_rewards_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        infos = locals_["info"]

        #add to buffers
        for key in self._reward_dict_temp.keys():
            self._reward_dict_temp[key].append(infos[key])

    def _log_final_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        infos = locals_["info"]
        if infos["TimeLimit.truncated"] or infos["is_success"]:
            for key in self._reward_dict_temp.keys():
                self._reward_dict[key].append(np.mean(self._reward_dict_temp[key]))
            for key in self._info_dict.keys():
                self._info_dict[key].append(infos[key])
            self._reward_dict["init_distance"].append(self.eval_env.get_attr("init_distance", 0)[0])
            self._reward_dict["init_perp_cosine_sim"].append(self.eval_env.get_attr("init_perp_cosine_sim", 0)[0])
            self._reward_dict["init_point_cosine_sim"].append(self.eval_env.get_attr("init_point_cosine_sim", 0)[0])
            loc = self.eval_env.get_attr("tree_goal_pos", 0)[0]
            self._reward_dict["pointx"].append(loc[0])
            self._reward_dict["pointy"].append(loc[1])
            self._reward_dict["pointz"].append(loc[2])

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._collisions_acceptable_buffer.append(self.eval_env.get_attr("collisions_acceptable", 0)[0])
        self._collisions_unacceptable_buffer.append(self.eval_env.get_attr("collisions_unacceptable", 0)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)
        self._log_final_metrics(_locals, _globals)

    def _init_buffer(self) -> None:
        self._is_success_buffer = {}
        self._is_success_buffer["is_success"] = []
        self._screens_buffer = []
        self._collisions_acceptable_buffer = []
        self._collisions_unacceptable_buffer = []

        self._reward_dict["movement_reward"] = []
        self._reward_dict["distance_reward"] = []
        self._reward_dict["terminate_reward"] = []
        self._reward_dict["collision_acceptable_reward"] = []
        self._reward_dict["collision_unacceptable_reward"] = []
        self._reward_dict["slack_reward"] = []
        self._reward_dict["condition_number_reward"] = []
        self._reward_dict["velocity_reward"] = []
        self._reward_dict["pointing_orientation_reward"] = []
        self._reward_dict["perpendicular_orientation_reward"] = []

        self._reward_dict["init_distance"] = []
        self._reward_dict["init_perp_cosine_sim"] = []
        self._reward_dict["init_point_cosine_sim"] = []
        self._reward_dict["pointx"] = []
        self._reward_dict["pointy"] = []
        self._reward_dict["pointz"] = []

        self._reward_dict_temp["movement_reward"] = []
        self._reward_dict_temp["distance_reward"] = []
        self._reward_dict_temp["terminate_reward"] = []
        self._reward_dict_temp["collision_acceptable_reward"] = []
        self._reward_dict_temp["collision_unacceptable_reward"] = []
        self._reward_dict_temp["slack_reward"] = []
        self._reward_dict_temp["condition_number_reward"] = []
        self._reward_dict_temp["velocity_reward"] = []
        self._reward_dict_temp["pointing_orientation_reward"] = []
        self._reward_dict_temp["perpendicular_orientation_reward"] = []

        self._info_dict["pointing_cosine_sim_error"] = []
        self._info_dict["perpendicular_cosine_sim_error"] = []
        self._info_dict["euclidean_error"] = []


    def eval_policy(self):

        # Reset success rate buffer
        self._init_buffer()
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

        print("Episode Rewards", episode_rewards)
        print(self._reward_dict)
        print(self._info_dict)
        print(self._is_success_buffer)
        reward_df = pd.DataFrame(self._reward_dict)
        success_df = pd.DataFrame(self._is_success_buffer)
        info_df = pd.DataFrame(self._info_dict)
        save_df = pd.concat([reward_df, success_df, info_df], axis=1)
        print(save_df, success_df)
        save_df.to_csv('reward.csv', index=False)
        # success_df.to_csv('success.csv', index=False)
        print("Success", np.mean(self._is_success_buffer["is_success"]))

parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())
parsed_args_dict = organize_args(parsed_args)
print(args)

if __name__ == "__main__":
    if parsed_args_dict['args_env']['use_optical_flow'] and parsed_args_dict['args_env']['optical_flow_subproc']:
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)
    add_arg_to_env('shared_var', shared_var, ['args_train', 'args_test', 'args_record'], parsed_args_dict)

    load_path_model = None
    load_path_mean_std = None
    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "../logs/{}/best_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "../logs/{}/mean_std.obj".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path = None

    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(device)
    eval_env = Monitor(PruningEnv(**args_test))
    eval_env.reset()
    assert load_path_mean_std
    assert load_path_model
    model = RecurrentPPOAE.load(load_path_model)
    model.policy.load_running_mean_std_from_file(load_path_mean_std)

    # evaluate_policy(model, eval_env, n_eval_episodes=1, render=False, deterministic=True)
    eval = CustomEvalCallback(eval_env, model, n_eval_episodes=2)#len(eval_env.trees[0].reachable_points))
    eval.eval_policy()