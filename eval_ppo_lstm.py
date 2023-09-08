
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
import multiprocessing as mp
from pruning_sb3.optical_flow import OpticalFlow


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

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._collisions_buffer.append(self.eval_env.get_attr("collisions", 0)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)

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

        print("Success", np.mean(self._is_success_buffer))


if __name__ == "__main__":
    manager = mp.Manager()
    # queue = multiprocessing.Queue()
    shared_dict = manager.dict()
    shared_queue = manager.Queue()
    shared_var = (shared_queue, shared_dict)
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=OpticalFlow, args=((224, 224), True, shared_var),
                          daemon=True)  # type: ignore[attr-defined]
    # pytype: enable=attribute-error
    process.start()

    # './meshes_and_urdf/urdf/trees/train',
    eval_env_kwargs =  {"renders" : True, "tree_urdf_path" :  './meshes_and_urdf/urdf/trees/envy/test', "tree_obj_path" : './meshes_and_urdf/meshes/trees/envy/test', "action_dim" :6,
                "maxSteps" : 300, "movement_reward_scale" : 1, "action_scale" :2, "distance_reward_scale" :0,
                "condition_reward_scale" :0, "terminate_reward_scale" : 5, "collision_reward_scale" : -0.01, 
                "slack_reward_scale" :-0.0001, "num_points" : 50, "orientation_reward_scale" : 2,  "name":"evalenv", "use_optical_flow": True, "shared_var": (shared_queue, shared_dict)}
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(device)
    eval_env = Monitor(PruningEnv(**eval_env_kwargs))
    load_path = "./logs/run/best_model.zip"
    model = RecurrentPPOAE.load(load_path)
    evaluate_policy(model, eval_env, n_eval_episodes=1, render=False, deterministic=True)
    # eval = CustomEvalCallback(eval_env, model)
    # eval.eval_policy()