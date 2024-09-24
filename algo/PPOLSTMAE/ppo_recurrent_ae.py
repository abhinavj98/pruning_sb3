import glob
import pickle
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional, Type, TypeVar, Union, List

import numpy as np
import torch as th
import torch.autograd
import torchvision
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
# from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

SelfRecurrentPPOAE = TypeVar("SelfRecurrentPPOAE", bound="RecurrentPPOAE")


# noinspection PyTupleAssignmentBalance,PyArgumentList
class RecurrentPPOAE(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
    #     "MlpLstmPolicy": MlpLstmPolicy,
    #     "CnnLstmPolicy": CnnLstmPolicy,
    #     "MultiInputLstmPolicy": MultiInputLstmPolicy,
    # }

    def __init__(
            self,
            policy: Union[str, Type[RecurrentActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            learning_rate_ae: Union[float, Schedule] = 3e-4,
            learning_rate_logstd: Union[float, Schedule] = 3e-4,
            n_steps: int = 128,
            batch_size: Optional[int] = 128,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.001,
            vf_coef: float = 0.5,
            ae_coeff: float = 0.,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,

    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.learning_rate_ae = learning_rate_ae
        self.learning_rate_logstd = learning_rate_logstd
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        print("Normalize advantage: ", self.normalize_advantage)
        self.target_kl = target_kl
        self._last_lstm_states = None
        self.ae_coeff = ae_coeff

        if _init_setup_model:
            self._setup_model()

    # noinspection PyTypeChecker
    def _setup_model(self) -> None:
        self._custom_setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space,
                                                              spaces.Dict) else RecurrentRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            lr_schedule_ae=self.lr_schedule_ae,
            lr_schedule_logstd=self.lr_schedule_logstd,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        # if not isinstance(self.policy, RecurrentActorCriticPolicy):
        #     raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.mse_loss = th.nn.MSELoss()

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        if self.verbose > 1:
            start_time = time.time()

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.verbose > 0:
            print("INFO: Collecting online rollout")

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        offline = False
        callback.update_locals(locals())
        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states,
                                                                              episode_starts)  # pylint: disable=unexpected-keyword-arg

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                        done_
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx: idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx: idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        # pylint: disable=unexpected-keyword-arg
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[
                            0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        if self.verbose > 1:
            print(f"DEBUG: Collected data in {time.time() - start_time:.2f}s")
        return True

    def _custom_update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        # TODO: Move to callback
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
        self.logger.record("train/learning_rate_ae", self.lr_schedule_ae(self._current_progress_remaining))
        if self.learning_rate_logstd is not None:
            self.logger.record("train/learning_rate_logstd", self.lr_schedule_logstd(self._current_progress_remaining))
            update_learning_rate(self.policy.optimizer_logstd,
                                 self.lr_schedule_logstd(self._current_progress_remaining))

        update_learning_rate(self.policy.optimizer, self.lr_schedule(self._current_progress_remaining))
        update_learning_rate(self.policy.optimizer_ae, self.lr_schedule_ae(self._current_progress_remaining))

    def _custom_setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_ae = get_schedule_fn(self.learning_rate_ae)
        if self.learning_rate_logstd is not None:
            self.lr_schedule_logstd = get_schedule_fn(self.learning_rate_logstd)
        else:
            self.lr_schedule_logstd = None

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._custom_update_learning_rate(
            [self.policy.optimizer, self.policy.optimizer_ae, self.policy.optimizer_logstd])

        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            ae_losses = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, depth_proxy, depth_proxy_recon = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                ae_l2_loss = self.mse_loss(depth_proxy, depth_proxy_recon)
                ae_losses.append(ae_l2_loss.item() * self.ae_coeff)
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + ae_l2_loss * self.ae_coeff

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step  
                self.policy.optimizer.zero_grad()
                self.policy.optimizer_ae.zero_grad()
                if self.learning_rate_logstd is not None:
                    self.policy.optimizer_logstd.zero_grad()

                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer_ae.step()
                if self.learning_rate_logstd is not None:
                    self.policy.optimizer_logstd.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        of_image_x = self.normalize_image(depth_proxy_recon[0, 0, :, :]).unsqueeze(0)
        of_image_y = self.normalize_image(depth_proxy_recon[0, 1, :, :]).unsqueeze(0)
        of_mask = depth_proxy_recon[0, 2, :, :].unsqueeze(0)

        # resize plot_img to be the same size as of_image
        depth_proxy_resized = F.interpolate(depth_proxy, size=(of_image_x.shape[1], of_image_x.shape[2]),
                                            mode='bilinear')
        plot_img_x = self.normalize_image(depth_proxy[0, 0, :, :]).unsqueeze(0)
        plot_img_y = self.normalize_image(depth_proxy[0, 1, :, :]).unsqueeze(0)
        plot_mask = depth_proxy[0, 2, :, :].unsqueeze(0)
        of_image_x_grid = torchvision.utils.make_grid(
            [of_image_x, plot_img_x])
        of_image_y_grid = torchvision.utils.make_grid(
            [of_image_y, plot_img_y])
        of_mask_grid = torchvision.utils.make_grid(
            [of_mask, plot_mask])

        self.logger.record("autoencoder/of_mask", Image(of_mask_grid, "CHW"), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("autoencoder/depth_proxy_x", Image(of_image_x_grid, "CHW"),
                           exclude=("stdout", "log", "json", "csv"))
        self.logger.record("autoencoder/depth_proxy_y", Image(of_image_y_grid, "CHW"),
                           exclude=("stdout", "log", "json", "csv"))
        self.logger.record("train/ae_loss", np.mean(ae_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def normalize_image(self, image):
        # Subtract by min and divide by max
        return (image - th.min(image.reshape(-1))) / (th.max(image.reshape(-1)) - th.min(image.reshape(-1)) + 1e-8)

    def learn(
            self: SelfRecurrentPPOAE,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "RecurrentPPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfRecurrentPPOAE:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


class TrajectoryIterableDataset(IterableDataset):
    """ Returns numpy observations from the dataset. Manually convert to torch tensors. """

    def __init__(self, file_list, observation_space):
        """
        :param file_list: List of all trajectory files
        :param observation_space: Observation space of the environment
        """
        self.file_list = file_list  # List of all trajectory files
        self.observation_space = observation_space

    def load_trajectory(self, file_path):
        """
        Loads a trajectory from a .pkl file.
        """
        with open(file_path, 'rb') as f:
            trajectory_data = pickle.load(f)
        return trajectory_data

    def trajectory_generator(self, trajectory_data):
        """
        Generator that yields each step from the trajectory.
        """
        for step in range(len(trajectory_data['observations'])):
            observation = trajectory_data['observations'][step]
            action = trajectory_data['actions'][step]
            reward = trajectory_data['rewards'][step]
            done = trajectory_data['dones'][step]
            next_observation = trajectory_data['next_observations'][step]

            # flattened_observation = self._flatten_obs(observation, self.observation_space)
            # flattened_next_observation = self._flatten_obs(next_observation, self.observation_space)

            # sample are numpy arrays
            # print(type(observation), type(action), type(reward), type(done), type(next_observation))
            sample = {
                'observation': observation,
                'action': action,
                'reward': reward,
                'done': done,
                'next_observation': next_observation,
            }
            yield sample

    def __iter__(self):
        """
        This method is called separately for each worker when using multiple workers.
        Each worker will randomly select files and process them.
        Make sure that num_workers is set to num_envs
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-worker case (or num_workers=0)
            file_list = self.file_list
        else:
            # Multi-worker case, each worker will have its own copy of the dataset and iterate independently
            worker_id = worker_info.id
            # Shuffle the file list for each worker (optional)
            file_list = self.file_list.copy()
            random.shuffle(file_list)

        # Loop through the files indefinitely (allows repetition)
        while True:
            for file_path in file_list:
                trajectory_data = self.load_trajectory(file_path)
                yield from self.trajectory_generator(trajectory_data)


class RecurrentPPOAEWithExpert(RecurrentPPOAE):
    """Allows use of data collected offline along with online data for training. The offline data is stored in a folder as pkl files."""

    def __init__(self, path_trajectories, use_online_data, use_offline_data, use_ppo_offline, use_online_bc, use_awac,
                 *args, **kwargs):
        if "_init_setup_model" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs, _init_setup_model=False)

        self.use_online_data = use_online_data
        self.use_offline_data = use_offline_data
        self.use_ppo_offline = use_ppo_offline  # Use both online and offline data
        self.use_online_bc = use_online_bc  # Use online data for behavior cloning
        self.use_awac = use_awac
        self.collect_online_data = self.use_online_data or self.use_ppo_offline or self.use_online_bc or self.use_awac
        self.collect_offline_data = self.use_offline_data or self.use_ppo_offline or self.use_awac or self.use_online_bc

        self.path_trajectories = path_trajectories

        # This is for loading purpose when model is not init but saved data is copied.
        _init_setup_model = kwargs.get("_init_setup_model", True)
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # self.expert_buffer = copy.deepcopy(self.rollout_buffer)  # Same structure as rollout buffer
        # No copy, only freedom
        self.num_expert_envs = self.n_envs
        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space,
                                                              spaces.Dict) else RecurrentRolloutBuffer
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.num_expert_envs, lstm.hidden_size)

        self.expert_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        expert_data = self.get_trajectory_files()  # Load file names only
        self.dataset = TrajectoryIterableDataset(expert_data, self.observation_space)
        self.dataloader = DataLoader(self.dataset, batch_size=self.num_expert_envs, num_workers=self.num_expert_envs,
                                     collate_fn=self.np_collate_fn)
        self.data_iter = iter(self.dataloader)

    @staticmethod
    def np_collate_fn(batch):
        """
        Custom collate function that keeps the data as NumPy arrays instead of converting to tensors.

        Args:
            batch (list): A list of samples from the dataset, where each sample is a dictionary
                          containing NumPy arrays (e.g., 'observation', 'action', 'reward', etc.).

        Returns:
            dict: A dictionary with batched NumPy arrays for each key in the sample dictionaries.
        """
        # Assuming each sample in the batch is a dictionary, and the keys are the same across samples
        batch_dict = {}

        # Collect keys from the first sample (assuming all samples have the same structure)
        first_sample = batch[0]

        for key in first_sample:
            # Stack NumPy arrays along a new axis to batch them together
            batch_dict[key] = np.stack([sample[key] for sample in batch], axis=0)  # Stack along batch dimension
            # else:
            #     # For non-NumPy data, keep them as lists
            #     batch_dict[key] = np.stack([np.array(sample[key]) for sample in batch], axis=0)  # Stack along batch dimension

        return batch_dict

    def get_trajectory_files(self):
        # Load expert data from expert_trajecotries folder. This is a list of pkl files
        # Each pkl file contains save_dict = {"tree_info": tree_info, "observations": observations, "actions": actions, "rewards": rewards, "dones": dones, "trajectory_in_frame": count_in_frame/len(actions)}
        # Load all the pkl files into a list
        expert_trajectories = glob.glob(self.path_trajectories + "/*.pkl")
        return list(expert_trajectories)

    def _flatten_obs(self, obs, observation_space):
        """
        Flatten a list of observations into a single dictionary.
        """
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in observation_space.spaces.keys()])

    def make_offline_rollouts(self, callback, expert_buffer: RolloutBuffer, n_rollout_steps) -> bool:
        # Make a list of offline observations, actions and trees
        if self.verbose > 0:
            print("INFO: Making offline rollouts")
        self.policy.set_training_mode(False)
        n_steps = 0
        expert_buffer.reset()
        # callback.update_locals(locals())
        # callback.on_rollout_start()

        # Sample expert episode
        self._last_episode_starts = np.ones((self.num_expert_envs,), dtype=bool)
        while n_steps < n_rollout_steps:
            try:
                # Get a batch of expert data from the DataLoader
                batch = next(self.data_iter)
            except StopIteration:
                # Reinitialize the iterator if we run out of data
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)

            # Unpack the batch
            # print(batch)
            # print(batch['observation'].keys())
            last_obs = self._flatten_obs(batch['observation'], self.observation_space)
            rewards = batch['reward']
            dones = batch['done']
            actions = batch['action']
            # print(type(rewards), type(dones), type(actions))
            # print([(k, type(v)) for k, v in last_obs.items()])
            self.num_timesteps += self.num_expert_envs
            n_steps += 1

            self._last_obs = last_obs
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions = obs_as_tensor(actions, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy.forward_expert(obs_tensor, self._last_lstm_states,
                                                                                     episode_starts, actions)

            actions = actions.cpu().numpy()

            expert_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states  # These get reset in forward_expert (process_sequence)

        next_obs = self._flatten_obs(batch['next_observation'],
                                     self.observation_space)  # Get the next observation to calculate the values
        # Dont increment expert_batch_idx
        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(next_obs, self.device), lstm_states.vf,
                                                episode_starts)  # pylint: disable=unexpected-keyword-arg
        expert_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # callback.on_rollout_end()
        # offline = False
        # callback.update_locals(locals())
        return True

    def train_online_batch(self, batch, clip_range, clip_range_vf):
        if self.verbose > 0:
            print("INFO: Training on rollout batch")
        # Train on online data
        actions = batch.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = batch.actions.long().flatten()

        # Convert mask from float to bool
        mask = batch.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        values, log_prob, entropy, depth_proxy, depth_proxy_recon = self.policy.evaluate_actions(
            batch.observations,
            actions,
            batch.lstm_states,
            batch.episode_starts,
        )

        values = values.flatten()
        # Normalize advantage
        advantages = batch.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - batch.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

        # Logging
        pg_loss = policy_loss.item()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = batch.old_values + th.clamp(
                values - batch.old_values, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        # Mask padded sequences
        ae_l2_loss = self.mse_loss(depth_proxy, depth_proxy_recon) * self.ae_coeff
        value_loss = th.mean(((batch.returns - values_pred) ** 2)[mask]) * self.vf_coef

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob[mask]) * self.ent_coef
        else:
            entropy_loss = -th.mean(entropy[mask]) * self.ent_coef

        online_loss = policy_loss + entropy_loss + value_loss + ae_l2_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = log_prob - batch.old_log_prob
            approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()

        return online_loss, pg_loss, clip_fraction, ae_l2_loss.item(), value_loss.item(), entropy_loss.item(), approx_kl_div, depth_proxy, depth_proxy_recon

    def train_online_bc(self, batch_online, batch_offline, clip_range, clip_range_vf):
        # Train on mix of online and offline data
        # Normalize advantage for both online and offline data together
        if self.verbose > 1:
            print("INFO: Training on mix batch")
        actions_online = batch_online.actions
        actions_offline = batch_offline.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions_online = batch_online.actions.long().flatten()
            actions_offline = batch_offline.actions.long().flatten()

        # Convert mask from float to bool
        mask_online = batch_online.mask > 1e-8
        mask_offline = batch_offline.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        values, log_prob, entropy, depth_proxy, depth_proxy_recon = self.policy.evaluate_actions(
            batch_online.observations,
            actions_online,
            batch_online.lstm_states,
            batch_online.episode_starts,
        )

        values = values.flatten()
        # Normalize advantage
        advantages = batch_online.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages[mask_online].mean()) / (advantages[mask_online].std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - batch_online.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask_online])

        # Logging
        pg_loss = policy_loss.item()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask_online]).item()

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = batch_online.old_values + th.clamp(
                values - batch_online.old_values, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        # Mask padded sequences
        ae_l2_loss = self.mse_loss(depth_proxy, depth_proxy_recon) * self.ae_coeff
        value_loss = th.mean(((batch_online.returns - values_pred) ** 2)[mask_online]) * self.vf_coef

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob[mask_online]) * self.ent_coef
        else:
            entropy_loss = -th.mean(entropy[mask_online]) * self.ent_coef

        online_loss = policy_loss + entropy_loss + value_loss + ae_l2_loss

        # behavior cloning loss  for offline data
        # Run policy on offline data
        _, log_prob_offline, _, _ = self.policy.forward_expert(
            batch_offline.observations,
            batch_offline.lstm_states,
            batch_offline.episode_starts,
            batch_offline.actions
        )

        log_prob_offline = th.clamp(log_prob_offline, -4, 4)
        bc_loss = -th.mean(log_prob_offline)
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = log_prob - batch_online.old_log_prob
            approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask_online]).cpu().numpy()

        return online_loss, bc_loss, pg_loss, clip_fraction, ae_l2_loss.item(), value_loss.item(), entropy_loss.item(), approx_kl_div, depth_proxy, depth_proxy_recon

    def train_ppo_offline(self, batch_online, batch_offline, clip_range, clip_range_vf):
        # Train on mix of online and offline data
        # Normalize advantage for both online and offline data together
        if self.verbose > 1:
            print("INFO: Training on mix batch")
        actions_online = batch_online.actions
        actions_offline = batch_offline.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions_online = batch_online.actions.long().flatten()
            actions_offline = batch_offline.actions.long().flatten()

        # Convert mask from float to bool
        mask_online = batch_online.mask > 1e-8
        mask_offline = batch_offline.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        values_online, log_prob_online, entropy_online, depth_proxy_online, depth_proxy_recon_online = self.policy.evaluate_actions(
            batch_online.observations,
            actions_online,
            batch_online.lstm_states,
            batch_online.episode_starts,
        )

        values_offline, log_prob_offline, entropy_offline, depth_proxy_offline, depth_proxy_recon_offline = self.policy.evaluate_actions(
            batch_offline.observations,
            actions_offline,
            batch_offline.lstm_states,
            batch_offline.episode_starts,
        )

        log_prob_offline = th.clamp(log_prob_offline, -4, 4)

        values_online = values_online.flatten()
        values_offline = values_offline.flatten()
        # Normalize advantage
        advantages_online = batch_online.advantages
        advantages_offline = batch_offline.advantages
        # Concatenate advantages
        advantages = th.cat((advantages_online, advantages_offline), 0)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_online = advantages[:len(advantages_online)]
        advantages_offline = advantages[len(advantages_online):]

        log_prob_expert = 0.  # Variance of 0.04
        # ratio between old and new policy, should be one at the first iteration
        ratio_current_old_online = th.exp(log_prob_online - batch_online.old_log_prob)
        ratio_current_old_offline = th.exp(log_prob_offline - batch_offline.old_log_prob)
        ratio_current_expert_offline = th.exp(
            log_prob_offline - log_prob_expert)  # Expert probability is 1, so log prob is 0
        ratio_old_expert_offline = th.exp(
            batch_offline.old_log_prob - log_prob_expert)  # Expert probability is 1, so log prob is 0

        # clipped surrogate loss for online
        policy_loss_1_online = advantages_online * ratio_current_old_online
        policy_loss_2_online = advantages_online * th.clamp(ratio_current_old_online, 1 - clip_range, 1 + clip_range)
        policy_loss_online = -th.mean(th.min(policy_loss_1_online, policy_loss_2_online)[mask_online])

        # clipped surrogate loss for offline
        # Create gaussian distribution

        policy_loss_1_offline = advantages_offline * th.clamp(ratio_current_expert_offline, 0.5, 2)
        policy_loss_2_offline = advantages_offline * th.clamp(ratio_current_old_offline, 1 - clip_range,
                                                              1 + clip_range) * th.clamp(ratio_old_expert_offline, 0.5,
                                                                                         2)
        policy_loss_offline = -th.mean(th.min(policy_loss_1_offline, policy_loss_2_offline)[mask_offline])

        # policy_loss_online = -th.mean(log_prob_online * th.exp(advantages_online/10))
        # policy_loss_offline = -th.mean(log_prob_offline * th.exp(advantages_offline/10))
        if self.clip_range_vf is None:
            # No clipping
            values_pred_online = values_online
            values_pred_offline = values_offline
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred_online = batch_online.old_values + th.clamp(
                values_online - batch_online.old_values, -clip_range_vf, clip_range_vf
            )
            values_pred_offline = batch_offline.old_values + th.clamp(
                values_offline - batch_offline.old_values, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        value_loss_online = th.mean(((batch_online.returns - values_pred_online) ** 2)[mask_online]) * self.vf_coef
        value_loss_offline = th.mean(
            (((batch_offline.returns - values_pred_offline) ** 2) * th.clamp(ratio_old_expert_offline, 0.5, 2))[
                mask_offline]) * self.vf_coef

        # Autoencoder loss
        ae_l2_loss_online = self.mse_loss(depth_proxy_online, depth_proxy_recon_online) * self.ae_coeff
        ae_l2_loss_offline = self.mse_loss(depth_proxy_offline, depth_proxy_recon_offline) * self.ae_coeff

        # Entropy loss favor exploration
        if entropy_online is None:
            # Approximate entropy when no analytical form
            entropy_loss_online = -th.mean(-log_prob_online[mask_online] + 1e-8) * self.ent_coef
        else:
            entropy_loss_online = -th.mean(entropy_online[mask_online]) * self.ent_coef

        if entropy_offline is None:
            # Approximate entropy when no analytical form
            entropy_loss_offline = -th.mean(-log_prob_offline[mask_offline] + 1e-8) * self.ent_coef
        else:
            entropy_loss_offline = -th.mean(entropy_offline[mask_offline]) * self.ent_coef

        online_loss = policy_loss_online + entropy_loss_online + value_loss_online + ae_l2_loss_online
        offline_loss = policy_loss_offline + entropy_loss_offline + value_loss_offline + ae_l2_loss_offline

        with th.no_grad():
            log_ratio_online = log_prob_online - batch_online.old_log_prob
            approx_kl_div_online = th.mean(
                ((th.exp(log_ratio_online) - 1) - log_ratio_online)[mask_online]).cpu().numpy()
            log_ratio_offline = log_prob_offline - batch_offline.old_log_prob
            approx_kl_div_offline = th.mean(
                ((th.exp(log_ratio_offline) - 1) - log_ratio_offline)[mask_offline]).cpu().numpy()

        clip_fraction_online = th.mean((th.abs(ratio_current_old_online - 1) > clip_range).float()[mask_online]).item()
        clip_fraction_offline = th.mean(
            (th.abs(ratio_current_old_offline - 1) > clip_range).float()[mask_offline]).item()

        ratio_current_old_online_mean = th.mean(ratio_current_old_online).item()
        ratio_current_old_offline_mean = th.mean(ratio_current_old_offline).item()
        ratio_old_expert_offline_mean = th.mean(ratio_old_expert_offline).item()
        log_prob_offline_mean = th.mean(log_prob_offline).item()
        log_prob_online_mean = th.mean(log_prob_online).item()
        advantages_online_mean = th.mean(advantages_online).item()
        advantages_offline_mean = th.mean(advantages_offline).item()

        offline_loss_dict = {"ratio_current_old": ratio_current_old_offline_mean,
                             "ratio_old_expert": ratio_old_expert_offline_mean,
                             "policy_loss": policy_loss_offline.item(),
                             "entropy_loss": entropy_loss_offline.item(), "value_loss": value_loss_offline.item(),
                             "ae_loss": ae_l2_loss_offline.item(), "approx_kl_div": approx_kl_div_offline,
                             "clip_fraction": clip_fraction_offline,
                             "advantages": advantages_offline_mean, "log_prob_offline": log_prob_offline_mean}
        online_loss_dict = {"ratio_current_old": ratio_current_old_online_mean,
                            "policy_loss": policy_loss_online.item(),
                            "entropy_loss": entropy_loss_online.item(), "value_loss": value_loss_online.item(),
                            "ae_loss": ae_l2_loss_online.item(), "approx_kl_div": approx_kl_div_online,
                            "clip_fraction": clip_fraction_online, "advantages": advantages_online_mean,
                            "log_prob_online": log_prob_online_mean}
        return online_loss, online_loss_dict, offline_loss, offline_loss_dict

    def train_awac(self, batch_online, batch_offline, clip_range, clip_range_vf):
        """Train using advantage weighted actor critic"""
        if self.verbose > 1:
            print("INFO: Training on mix batch")
        actions_online = batch_online.actions
        actions_offline = batch_offline.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions_online = batch_online.actions.long().flatten()
            actions_offline = batch_offline.actions.long().flatten()

        # Convert mask from float to bool
        mask_online = batch_online.mask > 1e-8
        mask_offline = batch_offline.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        values_online, log_prob_online, entropy_online, depth_proxy_online, depth_proxy_recon_online = self.policy.evaluate_actions(
            batch_online.observations,
            actions_online,
            batch_online.lstm_states,
            batch_online.episode_starts,
        )

        values_offline, log_prob_offline, entropy_offline, depth_proxy_offline, depth_proxy_recon_offline = self.policy.evaluate_actions(
            batch_offline.observations,
            actions_offline,
            batch_offline.lstm_states,
            batch_offline.episode_starts,
        )

        values_online = values_online.flatten()
        values_offline = values_offline.flatten()
        # Normalize advantage
        advantages_online = batch_online.advantages
        advantages_offline = batch_offline.advantages
        # Concatenate advantages
        advantages = th.cat((advantages_online, advantages_offline), 0)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_online = advantages[:len(advantages_online)]
        advantages_offline = advantages[len(advantages_online):]

        # values = th.cat((values_online, values_offline), 0)
        # returns = th.cat((batch_online.returns, batch_offline.returns), 0)
        # value_loss = th.mean(((returns - values) ** 2)) * self.vf_coef
        # log_probs = th.cat((log_prob_online, log_prob_offline), 0)
        # policy_loss = -th.mean(log_probs*th.exp(advantages/10))
        # loss_dict = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

        value_loss_online = th.mean(((batch_online.returns - values_online) ** 2)[mask_online]) * self.vf_coef
        value_loss_offline = th.mean(((batch_offline.returns - values_offline) ** 2)[mask_offline]) * self.vf_coef

        policy_loss_online = -th.mean(log_prob_online * th.exp(advantages_online / 5))
        policy_loss_offline = -th.mean(log_prob_offline * th.exp(advantages_offline / 5))

        entropy_loss_offline = -th.mean(-log_prob_offline) * self.ent_coef
        entropy_loss_online = -th.mean(-log_prob_online) * self.ent_coef

        loss_dict_online = {"policy_loss": policy_loss_online.item(), "value_loss": value_loss_online.item(),
                            "entropy_loss": entropy_loss_online.item(), "advantages": th.mean(advantages_online).item()}
        loss_dict_offline = {"policy_loss": policy_loss_offline.item(), "value_loss": value_loss_offline.item(),
                             "entropy_loss": entropy_loss_offline.item(),
                             "advantages": th.mean(advantages_offline).item()}
        online_loss = policy_loss_online + value_loss_online
        offline_loss = policy_loss_offline + value_loss_offline
        entropy_loss = entropy_loss_online + entropy_loss_offline

        return online_loss, loss_dict_online, offline_loss, loss_dict_offline, entropy_loss

    def train_offline_batch(self, batch, clip_range, clip_range_vf):
        actions = batch.actions

        if self.verbose > 1:
            print("INFO: Training on expert batch")
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = batch.actions.long().flatten()

        # Convert mask from float to bool
        mask = batch.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)
        # Check if any observations are nan
        for key, val in batch.observations.items():
            if th.isnan(val).any():
                print("Nan in observations")
        # assert not th.isnan(actions).any(), "Nan in actions"
        # assert not th.isnan(batch.lstm_states.pi).any(), "Nan in lstm pi states"
        # assert not th.isnan(batch.lstm_states.vf).any(), "Nan in lstm vf states"
        # assert not th.isnan(batch.episode_starts).any(), "Nan in episode starts"
        values, log_prob, entropy, depth_proxy, depth_proxy_recon = self.policy.evaluate_actions(
            batch.observations,
            actions,
            batch.lstm_states,
            batch.episode_starts,
        )

        values = values.flatten()
        # Normalize advantage
        advantages = batch.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

        # ratio between old and new online policy, should be one at the first iteration
        ratio_current_old = th.exp(log_prob - batch.old_log_prob)
        ratio_current_expert = th.exp(log_prob - 30)  # Expert probability is 1, so log prob is 0
        ratio_old_expert = th.exp(batch.old_log_prob - 30)  # Expert probability is 1, so log prob is 0

        # print("ratio_current_old", ratio_current_old, "log_prob", log_prob, "old_log_prob", rollout_data.old_log_prob)
        # print("ratio_current_expert", ratio_current_expert)
        # print("ratio_old_expert", ratio_old_expert)
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio_current_expert
        policy_loss_2 = advantages * th.clamp(ratio_current_old, 1 - clip_range, 1 + clip_range) * ratio_old_expert
        policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

        # Logging
        pg_loss = policy_loss.item()
        clip_fraction = th.mean((th.abs(ratio_current_old - 1) > clip_range).float()[mask]).item()

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = batch.old_values + th.clamp(
                values - batch.old_values, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        # Mask padded sequences
        ae_l2_loss = self.mse_loss(depth_proxy, depth_proxy_recon) * self.ae_coeff
        value_loss = th.mean(((batch.returns * ratio_old_expert - values_pred) ** 2)[mask]) * self.vf_coef

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob[mask]) * self.ent_coef
        else:
            entropy_loss = -th.mean(entropy[mask]) * self.ent_coef

        offline_loss = policy_loss + entropy_loss + value_loss + ae_l2_loss
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html

        with th.no_grad():
            log_ratio = log_prob - batch.old_log_prob
            approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()

        return offline_loss, pg_loss, clip_fraction, ae_l2_loss.item(), value_loss.item(), entropy_loss.item(), approx_kl_div, depth_proxy, depth_proxy_recon

    def train_expert(self, clip_range, clip_range_vf):
        self.policy.set_training_mode(True)
        # Optimization step
        self.policy.optimizer.zero_grad()
        self.policy.optimizer_ae.zero_grad()
        if self.learning_rate_logstd is not None:
            self.policy.optimizer_logstd.zero_grad()

        entropy_losses_online = []
        pg_losses_online, value_losses_online = [], []
        clip_fractions_online = []
        approx_kl_divs_online = []
        ae_losses_online = []
        online_losses = []
        advantages_online = []
        ratio_current_old_online = []

        entropy_losses_offline = []
        pg_losses_offline, value_losses_offline = [], []
        clip_fractions_offline = []
        approx_kl_divs_offline = []
        ae_losses_offline = []
        offline_losses = []
        advantages_offline = []
        ratio_current_old_offline = []
        ratio_old_expert_offline = []
        log_prob_offline = []
        log_prob_online = []

        online_data = None
        offline_data = None
        online_data_buffer = None
        offline_data_buffer = None

        loss_online = None
        loss_offline = None

        if self.use_online_bc:
            bc_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            if self.collect_offline_data:
                offline_data_buffer = self.expert_buffer.get(self.batch_size)
            if self.collect_online_data:
                online_data_buffer = self.rollout_buffer.get(self.batch_size)
            while True:
                try:
                    if self.collect_online_data:
                        offline_data = next(offline_data_buffer)
                    if self.collect_offline_data:
                        online_data = next(online_data_buffer)
                except StopIteration:
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                self.policy.optimizer_ae.zero_grad()
                if self.learning_rate_logstd is not None:
                    self.policy.optimizer_logstd.zero_grad()

                # Train the expert
                if self.use_offline_data:
                    (loss_offline, pg_loss_offline, clip_fraction_offline, ae_l2_loss_offline, value_loss_offline,
                     entropy_loss_offline, approx_kl_div_offline, depth_proxy,
                     depth_proxy_recon) = self.train_offline_batch(
                        offline_data, clip_range, clip_range_vf)
                    pg_losses_offline.append(pg_loss_offline)
                    clip_fractions_offline.append(clip_fraction_offline)
                    ae_losses_offline.append(ae_l2_loss_offline)
                    value_losses_offline.append(value_loss_offline)
                    entropy_losses_offline.append(entropy_loss_offline)
                    approx_kl_divs_offline.append(approx_kl_div_offline)

                    loss_offline.backward()

                elif self.use_online_data:
                    (loss_online, pg_loss_online, clip_fraction_online, ae_l2_loss_online, value_loss_online,
                     entropy_loss_online, approx_kl_div_online, depth_proxy,
                     depth_proxy_recon) = self.train_online_batch(
                        online_data, clip_range, clip_range_vf)
                    pg_losses_online.append(pg_loss_online)
                    clip_fractions_online.append(clip_fraction_online)
                    ae_losses_online.append(ae_l2_loss_online)
                    value_losses_online.append(value_loss_online)
                    entropy_losses_online.append(entropy_loss_online)
                    approx_kl_divs_online.append(approx_kl_div_online)

                    loss_online.backward()

                elif self.use_ppo_offline:
                    (loss_online, online_loss_dict, loss_offline, offline_loss_dict) = self.train_ppo_offline(
                        online_data, offline_data, clip_range, clip_range_vf)
                    pg_losses_online.append(online_loss_dict["policy_loss"])
                    clip_fractions_online.append(online_loss_dict["clip_fraction"])
                    ae_losses_online.append(online_loss_dict["ae_loss"])
                    value_losses_online.append(online_loss_dict["value_loss"])
                    entropy_losses_online.append(online_loss_dict["entropy_loss"])
                    approx_kl_divs_online.append(online_loss_dict["approx_kl_div"])
                    online_losses.append(loss_online.item())
                    advantages_online.append(online_loss_dict["advantages"])
                    ratio_current_old_online.append(online_loss_dict["ratio_current_old"])
                    log_prob_online.append(online_loss_dict["log_prob_online"])

                    pg_losses_offline.append(offline_loss_dict["policy_loss"])
                    clip_fractions_offline.append(offline_loss_dict["clip_fraction"])
                    ae_losses_offline.append(offline_loss_dict["ae_loss"])
                    value_losses_offline.append(offline_loss_dict["value_loss"])
                    entropy_losses_offline.append(offline_loss_dict["entropy_loss"])
                    approx_kl_divs_offline.append(offline_loss_dict["approx_kl_div"])
                    offline_losses.append(loss_offline.item())
                    advantages_offline.append(offline_loss_dict["advantages"])
                    ratio_current_old_offline.append(offline_loss_dict["ratio_current_old"])
                    ratio_old_expert_offline.append(offline_loss_dict["ratio_old_expert"])
                    log_prob_offline.append(offline_loss_dict["log_prob_offline"])

                elif self.use_online_bc:
                    (loss_online, bc_loss, pg_loss_online, clip_fraction_online, ae_l2_loss_online, value_loss_online,
                     entropy_loss_online, approx_kl_div_online, depth_proxy, depth_proxy_recon) = self.train_online_bc(
                        online_data, offline_data, clip_range, clip_range_vf)
                    pg_losses_online.append(pg_loss_online)
                    clip_fractions_online.append(clip_fraction_online)
                    ae_losses_online.append(ae_l2_loss_online)
                    value_losses_online.append(value_loss_online)
                    entropy_losses_online.append(entropy_loss_online)
                    approx_kl_divs_online.append(approx_kl_div_online)
                    online_losses.append(loss_online.item())
                    bc_losses.append(bc_loss.item())

                elif self.use_awac:
                    online_loss, loss_dict_online, offline_loss, loss_dict_offline, entropy_loss = self.train_awac(
                        online_data, offline_data, clip_range, clip_range_vf)
                    pg_losses_online.append(loss_dict_online["policy_loss"])
                    value_losses_online.append(loss_dict_online["value_loss"])
                    pg_losses_offline.append(loss_dict_offline["policy_loss"])
                    value_losses_offline.append(loss_dict_offline["value_loss"])
                    entropy_losses_online.append(loss_dict_online["entropy_loss"])
                    entropy_losses_offline.append(loss_dict_offline["entropy_loss"])
                    advantages_online.append(loss_dict_online["advantages"])
                    advantages_offline.append(loss_dict_offline["advantages"])

                    loss = online_loss + offline_loss + entropy_loss
                    loss.backward()

                # For BC loss remove variance from the optimization
                if self.use_online_bc:
                    offline_loss = bc_loss * 0.01
                    offline_loss.backward()
                    self.policy.optimizer_logstd.zero_grad()
                    online_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer_ae.step()
                    self.policy.optimizer_logstd.step()
                elif self.use_ppo_offline:
                    offline_loss = loss_offline / 2
                    online_loss = loss_online / 2
                    # loss = loss_offline/20+lose_online/2
                    # loss.backward()
                    # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    offline_loss.backward()
                    # self.policy.optimizer_logstd.zero_grad()
                    online_loss.backward()
                    # self.policy.optimizer.step()
                    # self.policy.optimizer_ae.step()
                    # self.policy.optimizer_logstd.zero_grad()
                    # self.policy.optimizer.zero_grad()
                    # self.policy.optimizer_ae.zero_grad()
                    # self.policy.optimizer_logstd.zero_grad()
                    # offline_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer_ae.step()
                    self.policy.optimizer_logstd.step()
                else:
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer_ae.step()
                    if self.learning_rate_logstd is not None:
                        self.policy.optimizer_logstd.step()

                if not continue_training:
                    break

            # print(self.rollout_buffer.values, self.rollout_buffer.returns)
        self._n_updates += self.n_epochs
        # assert not np.isnan(self.expert_buffer.values.flatten()).any(), "Nan in loss"
        # assert not np.isnan(self.expert_buffer.returns.flatten()).any(), "Nan in returns"
        # explained_var = explained_variance(self.expert_buffer.values.flatten(),
        #                                    self.expert_buffer.returns.flatten())

        explained_var_online = explained_variance(self.rollout_buffer.values.flatten(),
                                                  self.rollout_buffer.returns.flatten())
        explained_var_offline = explained_variance(self.expert_buffer.values.flatten(),
                                                   self.expert_buffer.returns.flatten())
        # Logs

        # of_image_x = self.normalize_image(depth_proxy_recon[0, 0, :, :]).unsqueeze(0)
        # of_image_y = self.normalize_image(depth_proxy_recon[0, 1, :, :]).unsqueeze(0)
        # of_mask = depth_proxy_recon[0, 2, :, :].unsqueeze(0)
        #
        # plot_img_x = self.normalize_image(depth_proxy[0, 0, :, :]).unsqueeze(0)
        # plot_img_y = self.normalize_image(depth_proxy[0, 1, :, :]).unsqueeze(0)
        # plot_mask = depth_proxy[0, 2, :, :].unsqueeze(0)
        # of_image_x_grid = torchvision.utils.make_grid(
        #     [of_image_x, plot_img_x])
        # of_image_y_grid = torchvision.utils.make_grid(
        #     [of_image_y, plot_img_y])
        # of_mask_grid = torchvision.utils.make_grid(
        #     [of_mask, plot_mask])
        #
        # self.logger.record("autoencoder/of_mask", Image(of_mask_grid, "CHW"),
        #                    exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("autoencoder/depth_proxy_x", Image(of_image_x_grid, "CHW"),
        #                    exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("autoencoder/depth_proxy_y", Image(of_image_y_grid, "CHW"),
        #                    exclude=("stdout", "log", "json", "csv"))

        if self.collect_online_data:
            self.logger.record("train_online/ae_loss", np.mean(ae_losses_online))
            self.logger.record("train_online/entropy_loss", np.mean(entropy_losses_online))
            self.logger.record("train_online/policy_gradient_loss", np.mean(pg_losses_online))
            self.logger.record("train_online/value_loss", np.mean(value_losses_online))
            self.logger.record("train_online/approx_kl", np.mean(approx_kl_divs_online))
            self.logger.record("train_online/clip_fraction", np.mean(clip_fractions_online))
            self.logger.record("train_online/loss", np.mean(online_losses))
            self.logger.record("train_online/explained_variance", explained_var_online)
            self.logger.record("train_online/advs", np.mean(advantages_online))
            self.logger.record("train_online/ratio_current_old", np.mean(ratio_current_old_online))
            self.logger.record("train_online/pred_values", np.mean(self.rollout_buffer.values).item())
            self.logger.record("train_online/returns", np.mean(self.rollout_buffer.returns))
            self.logger.record("train_online/log_prob", np.mean(log_prob_online))
            if hasattr(self.policy, "log_std"):
                self.logger.record("train_online/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train_online/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train_online/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train_online/clip_range_vf", clip_range_vf)
        if self.collect_offline_data:
            self.logger.record("train_offline/ae_loss", np.mean(ae_losses_offline))
            self.logger.record("train_offline/entropy_loss", np.mean(entropy_losses_offline))
            self.logger.record("train_offline/policy_gradient_loss", np.mean(pg_losses_offline))
            self.logger.record("train_offline/value_loss", np.mean(value_losses_offline))
            self.logger.record("train_offline/approx_kl", np.mean(approx_kl_divs_offline))
            self.logger.record("train_offline/clip_fraction", np.mean(clip_fractions_offline))
            self.logger.record("train_offline/loss", np.mean(offline_losses))
            self.logger.record("train_offline/explained_variance", explained_var_offline)
            self.logger.record("train_offline/advs", np.mean(advantages_offline))
            self.logger.record("train_offline/ratio_current_old", np.mean(ratio_current_old_offline))
            self.logger.record("train_offline/ratio_old_expert", np.mean(ratio_old_expert_offline))
            self.logger.record("train_offline/pred_values", np.mean(self.expert_buffer.values).item())
            self.logger.record("train_offline/returns",
                               np.mean(self.expert_buffer.returns).item())
            self.logger.record("train_offline/log_prob", np.mean(log_prob_offline))
            if self.use_online_bc:
                self.logger.record("train_offline/bc_loss", np.mean(bc_losses))

            if hasattr(self.policy, "log_std"):
                self.logger.record("train_offline/std", th.exp(self.policy.log_std).mean().item())

        return continue_training

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._custom_update_learning_rate(
            [self.policy.optimizer, self.policy.optimizer_ae, self.policy.optimizer_logstd])

        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        self.train_expert(clip_range, clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "RecurrentPPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfRecurrentPPOAE:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        continue_training = True
        print("Learning with bc", self.use_online_bc)
        print("Learning with online data", self.use_online_data)
        print("Learning with offline data", self.use_offline_data)
        print("Learning with mix data", self.use_ppo_offline)
        print("Learning with awac", self.use_awac)
        while self.num_timesteps < total_timesteps:
            if self.use_online_data or self.use_ppo_offline or self.use_online_bc or self.use_awac:
                # if self.num_timesteps > :
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                          n_rollout_steps=self.n_steps)
                # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                #                                           n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            if self.use_offline_data or self.use_ppo_offline or self.use_online_bc or self.use_awac:
                self.make_offline_rollouts(callback, self.expert_buffer, n_rollout_steps=self.n_steps)

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

        # TODO: Make this whole class compatible with the stable-baselines3 API

    @classmethod
    def load(  # noqa: C901
            cls,
            path,
            env: Optional[GymEnv] = None,
            path_expert_data: Optional[str] = None,
            use_online_data: bool = False,
            use_offline_data: bool = False,
            use_ppo_offline: bool = True,
            use_online_bc: bool = False,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
        import warnings
        from stable_baselines3.common.utils import (
            check_for_correct_spaces,
            get_system_info,
        )
        from stable_baselines3.common.vec_env.patch_gym import _convert_space
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"
        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(

            path_expert_data=path_expert_data,
            use_online_data=use_online_data,
            use_offline_data=use_offline_data,
            use_online_bc=use_online_bc,
            use_ppo_offline=use_ppo_offline,
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]  # pytype: disable=attribute-error
        return model
