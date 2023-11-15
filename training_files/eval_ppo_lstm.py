# Description: Evaluate the trained model
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
        self._reward_dict_temp["movement_reward"].append(infos["movement_reward"])
        self._reward_dict_temp["distance_reward"].append(infos["distance_reward"])
        self._reward_dict_temp["terminate_reward"].append(infos["terminate_reward"])
        self._reward_dict_temp["collision_reward"].append(infos["collision_reward"])
        self._reward_dict_temp["slack_reward"].append(infos["slack_reward"])
        self._reward_dict_temp["condition_number_reward"].append(infos["condition_number_reward"])
        self._reward_dict_temp["velocity_reward"].append(infos["velocity_reward"])
        self._reward_dict_temp["pointing_orientation_reward"].append(infos["pointing_orientation_reward"])
        self._reward_dict_temp["perpendicular_orientation_reward"].append(infos["perpendicular_orientation_reward"])

    def _log_final_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        infos = locals_["info"]
        if infos["TimeLimit.truncated"] or infos["is_success"]:
            self._reward_dict["pointing_cosine_sim_error"].append(infos["pointing_cosine_sim_error"])
            self._reward_dict["perpendicular_cosine_sim_error"].append(infos["perpendicular_cosine_sim_error"])
            self._reward_dict["euclidean_error"].append(infos["euclidean_error"])
            self._reward_dict["movement_reward"].append(np.mean(self._reward_dict_temp["movement_reward"]))
            self._reward_dict["distance_reward"].append(np.mean(self._reward_dict_temp["distance_reward"]))
            self._reward_dict["terminate_reward"].append(np.mean(self._reward_dict_temp["terminate_reward"]))
            self._reward_dict["collision_reward"].append(np.mean(self._reward_dict_temp["collision_reward"]))
            self._reward_dict["slack_reward"].append(np.mean(self._reward_dict_temp["slack_reward"]))
            self._reward_dict["condition_number_reward"].append(np.mean(self._reward_dict_temp["condition_number_reward"]))
            self._reward_dict["velocity_reward"].append(np.mean(self._reward_dict_temp["velocity_reward"]))
            self._reward_dict["pointing_orientation_reward"].append(np.mean(self._reward_dict_temp["pointing_orientation_reward"]))
            self._reward_dict["perpendicular_orientation_reward"].append(np.mean(self._reward_dict_temp["perpendicular_orientation_reward"]))
            self._reward_dict["init_distance"].append(self.eval_env.get_attr("init_distance", 0)[0])
            self._reward_dict["init_perp_cosine_sim"].append(self.eval_env.get_attr("init_perp_cosine_sim", 0)[0])
            self._reward_dict["init_point_cosine_sim"].append(self.eval_env.get_attr("init_point_cosine_sim", 0)[0])
            p_x = self.eval_env.get_attr("tree_goal_pos", 0)[0][0]
            p_y = self.eval_env.get_attr("tree_goal_pos", 0)[0][1]
            p_z = self.eval_env.get_attr("tree_goal_pos", 0)[0][2]
            self._reward_dict["pointx"].append(p_x)
            self._reward_dict["pointy"].append(p_y)
            self._reward_dict["pointz"].append(p_z)

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._collisions_buffer.append(self.eval_env.get_attr("collisions", 0)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)
        self._log_final_metrics(_locals, _globals)

    def eval_policy(self):

        # Reset success rate buffer
        self._is_success_buffer = {}
        self._is_success_buffer["is_success"] = []
        self._screens_buffer = []
        self._collisions_buffer = []
        self._reward_dict["movement_reward"] = []
        self._reward_dict["distance_reward"] = []
        self._reward_dict["terminate_reward"] = []
        self._reward_dict["collision_reward"] = []
        self._reward_dict["slack_reward"] = []
        self._reward_dict["condition_number_reward"] = []
        self._reward_dict["velocity_reward"] = []
        self._reward_dict["pointing_orientation_reward"] = []
        self._reward_dict["perpendicular_orientation_reward"] = []
        self._reward_dict["pointing_cosine_sim_error"] = []
        self._reward_dict["perpendicular_cosine_sim_error"] = []
        self._reward_dict["euclidean_error"] = []
        self._reward_dict["init_distance"] = []
        self._reward_dict["init_perp_cosine_sim"] = []
        self._reward_dict["init_point_cosine_sim"] = []
        self._reward_dict["pointx"] = []
        self._reward_dict["pointy"] = []
        self._reward_dict["pointz"] = []


        self._reward_dict_temp["movement_reward"] = []
        self._reward_dict_temp["distance_reward"] = []
        self._reward_dict_temp["terminate_reward"] = []
        self._reward_dict_temp["collision_reward"] = []
        self._reward_dict_temp["slack_reward"] = []
        self._reward_dict_temp["condition_number_reward"] = []
        self._reward_dict_temp["velocity_reward"] = []
        self._reward_dict_temp["pointing_orientation_reward"] = []
        self._reward_dict_temp["perpendicular_orientation_reward"] = []
        self._reward_dict_temp["pointing_cosine_sim_error"] = []
        self._reward_dict_temp["perpendicular_cosine_sim_error"] = []
        self._reward_dict_temp["euclidean_error"] = []


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
        print("Episode Rewards", episode_rewards)
        print(self._reward_dict)
        reward_df = pd.DataFrame(self._reward_dict)
        success_df = pd.DataFrame(self._is_success_buffer)
        save_df = pd.concat([reward_df, success_df], axis=1)
        print(save_df, success_df)
        save_df.to_csv('reward.csv', index=False)
        # success_df.to_csv('success.csv', index=False)
        print("Success", np.mean(self._is_success_buffer["is_success"]))

parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)
if __name__ == "__main__":

    optical_flow_subproc = False
    if args.USE_OPTICAL_FLOW and optical_flow_subproc:
        print("Using optical flow")
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)

    if args.LOAD_PATH:
        print("Loading model from {}".format(args.LOAD_PATH))
        load_path = "./logs/{}/best_model.zip".format(
            args.LOAD_PATH)  # ./nfs/stak/users/jainab/hpc-share/codes/pruning_sb3/logs/lowlr/best_model.zip"#Nonei
    else:
        load_path = None



    eval_env_kwargs = {"renders": args.RENDER, "tree_urdf_path": args.TREE_TEST_URDF_PATH,
                       "tree_obj_path": args.TREE_TEST_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                       "max_steps": args.EVAL_MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                       "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                       "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                       "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                       "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                       "slack_reward_scale": args.SLACK_REWARD_SCALE, "num_points":None,
                       "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                       "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                       "name": "evalenv", "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": optical_flow_subproc,
                       "shared_var": shared_var}
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(device)
    eval_env = Monitor(PruningEnv(**eval_env_kwargs))
    eval_env.reset()
    model = RecurrentPPOAE.load(load_path)
    # evaluate_policy(model, eval_env, n_eval_episodes=1, render=False, deterministic=True)
    eval = CustomEvalCallback(eval_env, model, n_eval_episodes=len(eval_env.trees[0].reachable_points))
    eval.eval_policy()