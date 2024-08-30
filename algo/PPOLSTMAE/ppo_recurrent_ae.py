import sys
import time
from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, List

import numpy as np
import pandas as pd
import torch as th
import torchvision
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F
from sb3_contrib.common.recurrent.type_aliases import RecurrentDictRolloutBufferSamples, RNNStates
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import glob
from torch.utils.data import DataLoader

SelfRecurrentPPOAE = TypeVar("SelfRecurrentPPOAE", bound="RecurrentPPOAE")


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

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

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
        :param n_steps: Number of experiences to collect per environment
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
        if self.verbose > 1:
            print("DEBUG: Collecting online rollout")

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
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

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


class ExpertDataset(Dataset):
    def __init__(self, expert_data, verbose = 0, load_from_disk=True):
        self.expert_data = expert_data
        self.load_from_disk = load_from_disk
        self.verbose = verbose

    def __len__(self):
        return len(self.expert_data)

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self.expert_data) - 1)
        if self.verbose > 1:
            start_time = time.time()
            print("DEBUG: Loading from disk", random_idx)
        if self.load_from_disk:
            with open(self.expert_data[random_idx], "rb") as f:
                data = pickle.load(f)
        else:
            data = self.expert_data[random_idx]
        if self.verbose> 1:
            print("DEBUG: Loaded from disk in", time.time() - start_time)
        return data


def collate_fn(batch):
    return batch
def create_expert_dataloader(expert_data, batch_size=1, load_from_disk=True, num_workers=1, verbose = 0):
    dataset = ExpertDataset(expert_data, verbose, load_from_disk)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

class RecurrentPPOAEWithExpert(RecurrentPPOAE):
    """Allows use of data collected offline along with online data for training. The offline data is stored in a folder as pkl files."""
    def __init__(self, path_expert_data, use_online_data, use_offline_data, mix_data,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_online_data = use_online_data
        self.use_offline_data = use_offline_data
        self.mix_data = mix_data #Use both online and offline data
        self.num_expert_envs = self.n_envs
        self.load_expert_from_disk = True

        # self.expert_buffer = copy.deepcopy(self.rollout_buffer)  # Same structure as rollout buffer
        # No copy, only freedom
        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space,
                                                              spaces.Dict) else RecurrentRolloutBuffer
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.num_expert_envs, lstm.hidden_size)

        self.expert_buffer  = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )


        self.path_expert_data = path_expert_data
        self.expert_data = self.load_expert_data()
        self.expert_batch_idx = np.zeros((self.num_expert_envs,), dtype=int)

        self.expert_batch = []
        #load batch size trajectories in self.expert_batch
        for i in range(self.num_expert_envs):
            self.expert_batch.append(self.get_expert_batch(1)[0])


    def get_expert_batch(self, num_traj):
        # Get a batch of expert data
        if self.load_expert_from_disk:
            expert_batch_files = random.choices(self.expert_data, k=num_traj)
            expert_batch = []
            for i, file in enumerate(expert_batch_files):
                with open(file, "rb") as f:
                    expert_batch.append(pickle.load(f))
        else:
            expert_batch = random.choices(self.expert_data, k=num_traj)

        return expert_batch

    def _flatten_obs(self, obs, observation_space):
        """
        asd
        """
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in observation_space.spaces.keys()])

    def load_expert_data(self):
        # Load expert data from expert_trajecotries folder. This is a list of pkl files
        # Each pkl file contains save_dict = {"tree_info": tree_info, "observations": observations, "actions": actions, "rewards": rewards, "dones": dones, "trajectory_in_frame": count_in_frame/len(actions)}
        # Load all the pkl files into a list
        expert_trajectories = glob.glob(self.path_expert_data + "/*.pkl")
        if self.load_expert_from_disk:
            expert_data = list(expert_trajectories)
        else:
            expert_data = []
            for expert_trajectory in expert_trajectories:
                print(expert_trajectory)
                with open(expert_trajectory, "rb") as f:
                    expert_data.append(pickle.load(f))  # These are on cpu
        return expert_data

    def step_expert(self):
        # Get expert data for each expert trajectory at timestep expert_batch_idx
        obs = []
        rewards = []
        dones = []
        actions = []
        infos = [{} for _ in range(len(self.expert_batch))]
        for i in range(len(self.expert_batch)):
            timestep = self.expert_batch_idx[i]
            obs.append(self.expert_batch[i]["observations"][timestep])
            rewards.append(self.expert_batch[i]["rewards"][timestep])
            dones.append(self.expert_batch[i]["dones"][timestep])
            actions.append(self.expert_batch[i]["actions"][timestep])
            if dones[-1]:
                self.expert_batch[i] = self.get_expert_batch(1)[0]
                self.expert_batch_idx[i] = 0
                infos[i] = {"terminal_observation": self.expert_batch[i]["last_obs"]}

        return self._flatten_obs(obs, self.env.observation_space), np.stack(rewards), np.stack(dones), np.stack(
            actions), infos

    def make_offline_rollouts(self, callback, expert_buffer: RolloutBuffer, n_rollout_steps) -> bool:
        # Make a list of offline observations, actions and trees
        offline = True
        if self.verbose > 1:
            print("DEBUG: Making offline rollouts")
        self.policy.set_training_mode(False)
        n_steps = 0
        expert_buffer.reset()
        # callback.update_locals(locals())
        # callback.on_rollout_start()

        # Sample expert episode
        self._last_episode_starts = np.ones((self.num_expert_envs,), dtype=bool)
        while n_steps < n_rollout_steps:

            last_obs, rewards, dones, actions, infos = self.step_expert()
            # Increment expert_batch_idx
            self.expert_batch_idx += 1

            # callback.update_locals(locals())
            # if callback.on_step() is False:
            #     return False

            self._update_info_buffer(infos)
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

            self.num_timesteps += self.num_expert_envs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states  # These get reset in forward_expert (process_sequence)

        last_obs, _, _, _, _ = self.step_expert()
        # Dont increment expert_batch_idx
        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(last_obs, self.device), lstm_states.vf, episode_starts)
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

    def train_mix_batch(self, batch_online, batch_offline, clip_range, clip_range_vf):
        # Train on mix of online and offline data
        # Normalize advantage for both online and offline data together
        if self.verbose > 0:
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
        #Concatenate advantages
        advantages = th.cat((advantages_online, advantages_offline), 0)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_online = advantages[:len(advantages_online)]
        advantages_offline = advantages[len(advantages_online):]

        log_prob_expert = 30 #Variance of 0.04
        # ratio between old and new policy, should be one at the first iteration
        ratio_current_old_online = th.exp(log_prob_online - batch_online.old_log_prob)
        ratio_current_old_offline = th.exp(log_prob_offline - batch_offline.old_log_prob)
        ratio_current_expert_offline = th.exp(log_prob_offline - log_prob_expert)  # Expert probability is 1, so log prob is 0
        ratio_old_expert_offline = th.exp(batch_offline.old_log_prob - log_prob_expert)  # Expert probability is 1, so log prob is 0


        # clipped surrogate loss for online
        policy_loss_1_online = advantages_online * ratio_current_old_online
        policy_loss_2_online = advantages_online * th.clamp(ratio_current_old_online, 1 - clip_range, 1 + clip_range)
        policy_loss_online = -th.mean(th.min(policy_loss_1_online, policy_loss_2_online)[mask_online])

        # clipped surrogate loss for offline
        #Create gaussian distribution

        policy_loss_1_offline = advantages_offline * ratio_current_expert_offline
        policy_loss_2_offline = advantages_offline * th.clamp(ratio_current_old_offline, 1 - clip_range, 1 + clip_range) * ratio_old_expert_offline
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
        value_loss_online = th.mean(((batch_online.returns*ratio_current_old_online - values_pred_online) ** 2)[mask_online]) * self.vf_coef
        value_loss_offline = th.mean(((batch_offline.returns*ratio_current_expert_offline - values_pred_offline) ** 2)[mask_offline]) * self.vf_coef

        #Autoencoder loss
        ae_l2_loss_online = self.mse_loss(depth_proxy_online, depth_proxy_recon_online) * self.ae_coeff
        ae_l2_loss_offline = self.mse_loss(depth_proxy_offline, depth_proxy_recon_offline) * self.ae_coeff

        # Entropy loss favor exploration
        if entropy_online is None:
            # Approximate entropy when no analytical form
            entropy_loss_online = -th.mean(-log_prob_online[mask_online]) * self.ent_coef
        else:
            entropy_loss_online = -th.mean(entropy_online[mask_online]) * self.ent_coef

        if entropy_offline is None:
            # Approximate entropy when no analytical form
            entropy_loss_offline = -th.mean(-log_prob_offline[mask_offline]) * self.ent_coef
        else:
            entropy_loss_offline = -th.mean(entropy_offline[mask_offline]) * self.ent_coef

        online_loss = policy_loss_online + entropy_loss_online + value_loss_online + ae_l2_loss_online
        offline_loss = policy_loss_offline + entropy_loss_offline + value_loss_offline + ae_l2_loss_offline

        with th.no_grad():
            log_ratio_online = log_prob_online - batch_online.old_log_prob
            approx_kl_div_online = th.mean(((th.exp(log_ratio_online) - 1) - log_ratio_online)[mask_online]).cpu().numpy()
            log_ratio_offline = log_prob_offline - batch_offline.old_log_prob
            approx_kl_div_offline = th.mean(((th.exp(log_ratio_offline) - 1) - log_ratio_offline)[mask_offline]).cpu().numpy()


        clip_fraction_online = th.mean((th.abs(ratio_current_old_online - 1) > clip_range).float()[mask_online]).item()
        clip_fraction_offline = th.mean((th.abs(ratio_current_old_offline - 1) > clip_range).float()[mask_offline]).item()

        ratio_current_old_online_mean = th.mean(ratio_current_old_online).detach().cpu().numpy()
        ratio_current_old_offline_mean = th.mean(ratio_current_old_offline).detach().cpu().numpy()
        ratio_old_expert_offline_mean = th.mean(ratio_old_expert_offline).detach().cpu().numpy()


        offline_loss_dict = {"ratio_current_old": ratio_current_old_offline_mean, "ratio_old_expert": ratio_old_expert_offline_mean, "policy_loss": policy_loss_offline.item(), "entropy_loss": entropy_loss_offline.item(), "value_loss": value_loss_offline.item(), "ae_loss": ae_l2_loss_offline.item(), "approx_kl_div": approx_kl_div_offline, "clip_fraction": clip_fraction_offline, "advantages": th.mean(advantages_offline).cpu().numpy()}
        online_loss_dict =  {"ratio_current_old" : ratio_current_old_online_mean, "policy_loss": policy_loss_online.item(), "entropy_loss": entropy_loss_online.item(), "value_loss": value_loss_online.item(), "ae_loss": ae_l2_loss_online.item(), "approx_kl_div": approx_kl_div_online, "clip_fraction": clip_fraction_online, "advantages": th.mean(advantages_online).cpu().numpy()}
        return online_loss, online_loss_dict, offline_loss, offline_loss_dict

    def train_offline_batch(self, batch, clip_range, clip_range_vf):

        actions = batch.actions

        if self.verbose > 0:
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
        ratio_current_expert = th.exp(log_prob - 0)  # Expert probability is 1, so log prob is 0
        ratio_old_expert = th.exp(batch.old_log_prob - 0)  # Expert probability is 1, so log prob is 0

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
        value_loss = th.mean(((batch.returns - values_pred) ** 2)[mask]) * self.vf_coef

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

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            if self.use_offline_data or self.mix_data:
                offline_data_buffer = self.expert_buffer.get(self.batch_size)
            if self.use_online_data or self.mix_data:
                online_data_buffer = self.rollout_buffer.get(self.batch_size)
            while True:
                try:
                    if self.use_offline_data or self.mix_data:
                        offline_data = next(offline_data_buffer)
                    if self.use_online_data or self.mix_data:
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
                     entropy_loss_offline, approx_kl_div_offline, depth_proxy, depth_proxy_recon) = self.train_offline_batch(
                        offline_data, clip_range, clip_range_vf)
                    pg_losses_offline.append(pg_loss_offline)
                    clip_fractions_offline.append(clip_fraction_offline)
                    ae_losses_offline.append(ae_l2_loss_offline)
                    value_losses_offline.append(value_loss_offline)
                    entropy_losses_offline.append(entropy_loss_offline)
                    approx_kl_divs_offline.append(approx_kl_div_offline)


                    # if self.target_kl is not None and approx_kl_div_offline > 1.5 * self.target_kl:
                    #     continue_training = False
                    #     if self.verbose >= 1:
                    #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    #     break
                    loss_offline.backward()

                if self.use_online_data:
                    (loss_online, pg_loss_online, clip_fraction_online, ae_l2_loss_online, value_loss_online,
                     entropy_loss_online, approx_kl_div_online, depth_proxy, depth_proxy_recon) = self.train_online_batch(
                        online_data, clip_range, clip_range_vf)
                    pg_losses_online.append(pg_loss_online)
                    clip_fractions_online.append(clip_fraction_online)
                    ae_losses_online.append(ae_l2_loss_online)
                    value_losses_online.append(value_loss_online)
                    entropy_losses_online.append(entropy_loss_online)
                    approx_kl_divs_online.append(approx_kl_div_online)

                    # if self.target_kl is not None and approx_kl_div_online > 1.5 * self.target_kl:
                    #     continue_training = False
                    #     if self.verbose >= 1:
                    #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    #     break

                    loss_online.backward()

                if self.mix_data:


                    (loss_online, online_loss_dict, loss_offline, offline_loss_dict) = self.train_mix_batch(
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

                    loss = (loss_online + loss_offline)/2
                    loss.backward()
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

        explained_var_online = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        explained_var_offline = explained_variance(self.expert_buffer.values.flatten(), self.expert_buffer.returns.flatten())
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

        if self.use_online_data or self.mix_data:
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

            if hasattr(self.policy, "log_std"):
                self.logger.record("train_online/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train_online/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train_online/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train_online/clip_range_vf", clip_range_vf)
        if self.use_offline_data or self.mix_data:
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

            if hasattr(self.policy, "log_std"):
                self.logger.record("train_offline/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train_offline/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train_offline/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train_offline/clip_range_vf", clip_range_vf)
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
        while self.num_timesteps < total_timesteps:
            if self.use_online_data or self.mix_data:
                # if self.num_timesteps > :
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)
                # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                #                                           n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            if self.use_offline_data or self.mix_data:
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
