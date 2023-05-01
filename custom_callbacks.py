
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CallbackList
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import cv2
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

class CustomTrainCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomTrainCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        self.logger = None  # stable_baselines3.common.logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        return True

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        #log episode


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        infos = self.locals["infos"]
        self.logger.record("rollout/movement_reward", infos[0]["movement_reward"])
        self.logger.record("rollout/distance_reward", infos[0]["distance_reward"])
        self.logger.record("rollout/terminate_reward", infos[0]["terminate_reward"])
        self.logger.record("rollout/collision_reward", infos[0]["collision_reward"])
        self.logger.record("rollout/slack_reward", infos[0]["slack_reward"])
        self.logger.record("rollout/condition_number_reward", infos[0]["condition_number_reward"])
        self.logger.record("rollout/velocity_reward", infos[0]["velocity_reward"])
        # self.logger.record("train/singularit_terminated", info["total_reward"])

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.

        """
        pass


# class VideoRecorderCallback(BaseCallback):
#     def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
#         """
#         Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

#         :param eval_env: A gym environment from which the trajectory is recorded
#         :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
#         :param n_eval_episodes: Number of episodes to render
#         :param deterministic: Whether to use deterministic or stochastic policy
#         """
#         super().__init__()
#         self.sum_collisions = 0
#         self._eval_env = eval_env
#         self._render_freq = render_freq
#         self._n_eval_episodes = n_eval_episodes
#         self._deterministic = deterministic

#     def _on_step(self) -> bool:
#         if self.n_calls % self._render_freq == 0:
#             screens = []

#             # def get_action_critic(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#             #     print(_locals)
                
#             def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#                 """
#                 Renders the environment in its current state, recording the screen in the captured `screens` list

#                 :param _locals: A dictionary containing all local variables of the callback's scope
#                 :param _globals: A dictionary containing all global variables of the callback's scope
#                 """
#                 screen = self._eval_env.render()
#                 # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                
#                 # critic_value = ppo.policy.critic(memory.depth_features[-1].unsqueeze(0), memory.states[-1])
#                 # debug_img = cv2.putText(debug_img, "Critic: "+str(critic_value.item()), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 
#                 #     1, (255,0,0), 2, cv2.LINE_AA)
#                 screen = cv2.putText(screen, "Reward: "+str(_locals['reward']), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, (255,0,0), 2, cv2.LINE_AA)
#                 screen = cv2.putText(screen, "Action: "+str(self._eval_env.rev_actions[int(_locals['actions'])]), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, (255,0,0), 2, cv2.LINE_AA)
#                 screen = cv2.putText(screen, "Current: "+str(self._eval_env.achieved_goal), (0,140), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, (255,0,0), 2, cv2.LINE_AA)
#                 screen = cv2.putText(screen, "Goal: "+str(self._eval_env.desired_goal), (0,170), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, (255,0,0), 2, cv2.LINE_AA)
#                 screens.append(screen.transpose(2, 0, 1))

#                 def log_collisions(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
#                     if _locals['done']:
#                         self.sum_collisions += self._eval_env.collisions
#                         print(self.sum_collisions)

#                 mean_reward, std_reward = evaluate_policy(
#                     self.model,
#                     self._eval_env,
#                     callback=[grab_screens, log_collisions],
#                     n_eval_episodes=self._n_eval_episodes,
#                     deterministic=self._deterministic,
#                 )

#                 self.logger.record(
#                     "eval/video",
#                     Video(th.ByteTensor(np.array([screens])), fps=10),
#                     exclude=("stdout", "log", "json", "csv"),
#                 )
#                 self.logger.record(
#                     "eval/collisions",
#                     self.sum_collisions
#                 )
                
#         return True


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
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        #For video
        self._screens_buffer = []
        self._collisions_buffer = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

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
        self.logger.record("eval/movement_reward", infos["movement_reward"])
        self.logger.record("eval/distance_reward", infos["distance_reward"])
        self.logger.record("eval/terminate_reward", infos["terminate_reward"])
        self.logger.record("eval/collision_reward", infos["collision_reward"])
        self.logger.record("eval/slack_reward", infos["slack_reward"])
        self.logger.record("eval/condition_number_reward", infos["condition_number_reward"])
        self.logger.record("eval/velocity_reward", infos["velocity_reward"])

    def _grab_screen_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        """
        Renders the environment in its current state, recording the screen in the captured `screens` list

        :param _locals: A dictionary containing all local variables of the callback's scope
        :param _globals: A dictionary containing all global variables of the callback's scope
        """
        episode_counts = _locals["episode_counts"][0]
        if episode_counts == 0:
            screen = self.eval_env.render()
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            
            # critic_value = ppo.policy.critic(memory.depth_features[-1].unsqueeze(0), memory.states[-1])
            # debug_img = cv2.putText(debug_img, "Critic: "+str(critic_value.item()), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 
            #     1, (255,0,0), 2, cv2.LINE_AA)
            screen = cv2.putText(screen, "Reward: "+str(_locals['reward']), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255,0,0), 2, cv2.LINE_AA)
            screen = cv2.putText(screen, "Action: "+" ".join(str(x) for x in _locals['actions']), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
            screen = cv2.putText(screen, "Current: "+str(self.eval_env.get_attr("achieved_goal", 0)[0]), (0,140), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255,0,0), 2, cv2.LINE_AA)
            screen = cv2.putText(screen, "Goal: "+str(self.eval_env.get_attr("desired_goal", 0)[0]), (0,170), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255,0,0), 2, cv2.LINE_AA)
            screen = cv2.putText(screen, "J_velocity: "+" ".join(str(x) for x in self.eval_env.get_attr("joint_velocities", 0)[0]), (0,200), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255,0,0), 2, cv2.LINE_AA) #str(_locals['actions'])
            self._screens_buffer.append(screen.transpose(2, 0, 1))

    def _log_collisions(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._collisions_buffer.append(self.eval_env.get_attr("collisions", 0)[0])

    def _master_callback(self, _locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        self._grab_screen_callback(_locals, _globals)
        self._log_collisions(_locals, _globals)
        self._log_success_callback(_locals, _globals)
        self._log_rewards_callback(_locals, _globals)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self._screens_buffer = []
            self._collisions_buffer = []
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

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_collisions = np.sum(self._collisions_buffer)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record(
                    "eval/video",
                    Video(th.ByteTensor(np.array([self._screens_buffer])), fps=10),
                    exclude=("stdout", "log", "json", "csv"),
                )
            self.logger.record("eval/collisions", mean_collisions)
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    # def update_child_locals(self, locals_: Dict[str, Any]) -> None:
    #     """
    #     Update the references to the local variables.
    #     :param locals_: the local variables during rollout collection
    #     """
    #     if self.callback:
    #         self.callback.update_locals(locals_)
