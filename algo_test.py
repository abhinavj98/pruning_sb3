from tabnanny import verbose
import gym
from a2c import A2CWithAE
from a2c_with_ae_policy import ActorCriticWithAePolicy
from ppo_ae import PPOAE
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
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.model.get_env().logger = self.logger
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
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.

        """
        pass
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video



class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            # def get_action_critic(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            #     print(_locals)
                
            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                
                # critic_value = ppo.policy.critic(memory.depth_features[-1].unsqueeze(0), memory.states[-1])
                # debug_img = cv2.putText(debug_img, "Critic: "+str(critic_value.item()), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 
                #     1, (255,0,0), 2, cv2.LINE_AA)
                screen = cv2.putText(screen, "Reward: "+str(_locals['reward']), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                screen = cv2.putText(screen, "Action: "+str(self._eval_env.rev_actions[int(_locals['actions'])]), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                screen = cv2.putText(screen, "Current: "+str(self._eval_env.achieved_goal), (0,140), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                screen = cv2.putText(screen, "Goal: "+str(self._eval_env.desired_goal), (0,170), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                screens.append(screen.transpose(2, 0, 1))

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "eval/vreward", mean_reward
            )
            self.logger.record(
                "eval/vreward_std", std_reward
            )
            self.logger.record(
                "eval/video",
                Video(th.ByteTensor(np.array([screens])), fps=10),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

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
# set up logger


        # Create eval callback if needed
env = ur5GymEnv(renders=False)
# eval_env = ur5GymEnv(renders=False, eval=True)
new_logger = utils.configure_logger(verbose = 0, tensorboard_log = "./runs/", reset_num_timesteps = True)
env.logger = new_logger 
eval_env = ur5GymEnv(renders=False, name = "evalenv")
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=1000,
                             deterministic=True, render=False)
# env = DummyVecEnv([lambda: env])
# eval_env = DummyVecEnv([lambda: eval_env])
#print(env.action_space)
# It will check your custom environment and output additional warnings if needed
# check_env(env)
video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
a = CustomCallback()
policy_kwargs = {
        "actor_class":  Actor,
        "critic_class":  Critic,
        "actor_kwargs": {"state_dim": 32+10*3, "emb_size":128, "action_dim":12, "action_std":1},
        "critic_kwargs": {"state_dim": 32+10*3, "emb_size":128, "action_dim":1, "action_std":1},
        "features_extractor_class" : AutoEncoder,
        "optimizer_class" : th.optim.Adam
        }#ActorCriticWithAePolicy(env.observation_space, env.action_space, linear_schedule(0.001), Actor(None, 128*7*7+10*3,128, 12, 1 ), Critic(None, 128*7*7+10*3, 128,1,1), features_extractor_class =  AutoEncoder)
model = PPOAE(ActorCriticWithAePolicy, env, policy_kwargs=policy_kwargs, learning_rate=0.001)
model.set_logger(new_logger)
print("Using device: ", utils.get_device())

env.reset()
for _ in range(1000):
    # env.render() 
    env.step(env.action_space.sample()) # take a random action
env.reset()
model.learn(1000000, callback=[video_recorder, a, eval_callback], progress_bar = True)
