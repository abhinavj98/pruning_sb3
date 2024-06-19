import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pruning_sb3')))

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
    SquashedDiagGaussianDistribution
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import BasePolicy
from pruning_sb3.pruning_gym.running_mean_std import RunningMeanStd
import pickle
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from torchvision.transforms import functional as F

#Optical flow
from pruning_sb3.pruning_gym.optical_flow import OpticalFlow

class ActorCriticPolicySquashed(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        # Replace action distribution with a Squashed
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(action_space))  # , self.full_std)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist,
                          StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
        print("LOG STD INIT: ", self.log_std_init)
        print("LOG STD: ", self.log_std)
        # multiply action net weight by 10
        # self.action_net.weight.data *= 10
        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # else:
        #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std)
        # elif isinstance(self.action_dist, CategoricalDistribution):
        #     # Here mean_actions are the logits before the softmax
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, MultiCategoricalDistribution):
        #     # Here mean_actions are the flattened logits
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, BernoulliDistribution):
        #     # Here mean_actions are the logits (before rounding to get the binary actions)
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        # else:
        #     raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class RecurrentActorCriticPolicy(ActorCriticPolicySquashed):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            lr_schedule_ae: Schedule = 0.0001,
            lr_schedule_logstd: Schedule = 0.0001,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lstm_hidden_size: int = 256,
            n_lstm_layers: int = 1,
            shared_lstm: bool = False,
            enable_critic_lstm: bool = True,
            lstm_kwargs: Optional[Dict[str, Any]] = None,
            features_dim_critic_add: Optional[int] = None,
            use_optical_flow = True,
            algo_size = (224, 224)
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.features_dim_critic_add = features_dim_critic_add
        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.use_optical_flow = use_optical_flow
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        self.algo_size = algo_size
        assert not (
                self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
                self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)
        if self.use_optical_flow:
            self.optical_flow_model = OpticalFlow(size = self.algo_size)
        # Use a separate LSTM for the critic
        # TODO: TEST
        if self.enable_critic_lstm:
            features_dim = self.features_dim if features_dim_critic_add is None else self.features_dim + self.features_dim_critic_add
            self.lstm_critic = nn.LSTM(
                features_dim,  # critic features dim
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
        # num_channels = self.features_extractor.in_channels
        self.running_mean_var_oflow_x = RunningMeanStd(shape=(1,))
        self.running_mean_var_oflow_y = RunningMeanStd(shape=(1,))

        # Setup optimizer with initial learning rate
        if lr_schedule_logstd is not None:
            self.optimizer_logstd = self.optimizer_class([self.log_std], lr=lr_schedule_logstd(1),
                                                         **self.optimizer_kwargs)
            self.optimizer = self.optimizer_class(
                [*self.lstm_actor.parameters(), *self.lstm_critic.parameters(), *self.value_net.parameters(),
                 *self.action_net.parameters()], lr=lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = self.optimizer_class(
                [*self.lstm_actor.parameters(), *self.lstm_critic.parameters(), *self.value_net.parameters(),
                 *self.action_net.parameters(), self.log_std], lr=lr_schedule(1), **self.optimizer_kwargs)
            self.optimizer_logstd = None
        if lr_schedule_ae is not None:
            self.optimizer_ae = self.optimizer_class(self.features_extractor.parameters(), lr=lr_schedule_ae(1),
                                                     **self.optimizer_kwargs)

    def _normalize_using_running_mean_std(self, x, mean_std_tuple):
        mean_x = mean_std_tuple[0].mean.to(self.device, dtype=th.float32)
        std_x = th.sqrt(mean_std_tuple[0].var).to(self.device, dtype=th.float32)
        mean_y = mean_std_tuple[1].mean.to(self.device, dtype=th.float32)
        std_y = th.sqrt(mean_std_tuple[1].var).to(self.device, dtype=th.float32)
        normalize_array = copy.deepcopy(x)
        normalize_array[:, 0, :, :] = (x[:, 0, :, :] - mean_x) / (std_x + 1e-8)
        normalize_array[:, 1, :, :] = (x[:, 1, :, :] - mean_y) / (std_y + 1e-8)
        return normalize_array

    def _unnormalize_using_running_mean_std(self, x, mean_std_tuple):
        mean_x = mean_std_tuple[0].mean.to(self.device, dtype=th.float32)
        std_x = th.sqrt(mean_std_tuple[0].var).to(self.device, dtype=th.float32)
        mean_y = mean_std_tuple[1].mean.to(self.device, dtype=th.float32)
        std_y = th.sqrt(mean_std_tuple[1].var).to(self.device, dtype=th.float32)
        x[:, 0, :, :] = (x[:, 0, :, :] * (std_x + 1e-8)) + mean_x
        x[:, 1, :, :] = (x[:, 1, :, :] * (std_y + 1e-8)) + mean_y
        return x

    # Load running_mean_std from pkl file
    class CPU_Unpickler(pickle.Unpickler):

        def find_class(self, module, name):
            import io
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: th.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)
    def load_running_mean_std_from_file(self, path):
        with open(path, 'rb') as f:
            if sys.platform == 'darwin':
                print("MAC")
                self.running_mean_var_oflow_x, self.running_mean_var_oflow_y = RecurrentActorCriticPolicy.CPU_Unpickler(f).load()
            else:
                print("NOT MAC")
                self.running_mean_var_oflow_x, self.running_mean_var_oflow_y = pickle.load(f)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
            features: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
            lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous cell and hidden states of the LSTM
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(
            self,
            obs: th.Tensor,
            lstm_states: RNNStates,
            episode_starts: th.Tensor,
            deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features, _, _ = self.extract_features(obs)

        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts,
                                                               self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_depth_proxy(self, rgb, prev_rgb, point_mask):
        optical_flow = self.optical_flow_model.calculate_optical_flow(rgb, prev_rgb)
        depth_proxy = th.cat((optical_flow, point_mask), dim = 1)

        return depth_proxy

    def extract_features(self, obs: th.Tensor):  # -> tuple[th.Tensor, th.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        # preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        # Add running mean and var
        # TODO: compute depth proxy here
        # from PIL import Image
        # import numpy as np
        # import time
        # #resize obs['rgb']
        # rgb = F.resize(obs['rgb'], size = (512, 512))
        # depth_proxy_resize = F.resize(depth_proxy, size = (512, 512))
        # optical_flow_x = depth_proxy_resize[:, 0, :, :]
        # optical_flow_y = depth_proxy_resize[:, 1, :, :]
        # #comvert to rgb
        # # optical_flow_x = (optical_flow_x - optical_flow_x.min()) / (optical_flow_x.max() - optical_flow_x.min())
        # # optical_flow_y = (optical_flow_y - optical_flow_y.min()) / (optical_flow_y.max() - optical_flow_y.min())
        # # optical_flow_x = (optical_flow_x * 255)#.cpu().detach().numpy()
        # # optical_flow_y = (optical_flow_y * 255)#.cpu().detach().numpy()
        # # optical_flow_x = optical_flow_x.astype(np.uint8)
        # # optical_flow_y = optical_flow_y.astype(np.uint8)
        # #Make 3 channel
        # # optical_flow_x = th.stack((optical_flow_x, optical_flow_x, optical_flow_x), axis = 1)
        # # optical_flow_y = th.stack((optical_flow_y, optical_flow_y, optical_flow_y), axis = 1)
        # from torchvision.utils import flow_to_image
        # from PIL import Image
        # import time
        # flow = th.cat((optical_flow_x, optical_flow_y), dim = 0)
        #
        # flow_imgs = flow_to_image(flow).unsqueeze(0)
        # # print(optical_flow_y.shape, optical_flow_x.shape)
        # # optical_flow_x = Image.fromarray(optical_flow_x)
        # # optical_flow_y = Image.fromarray(optical_flow_y)
        # # print(rgb.shape, optical_flow_x.shape, optical_flow_y.shape)
        # save_img = th.cat((rgb, flow_imgs), dim = 3)
        # # print(save_img.shape)
        # #Save image
        # save_img = save_img[0].permute(1, 2, 0).cpu().detach().numpy()
        # # save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min())
        # save_img = (save_img * 255).astype(np.uint8)
        # save_img = Image.fromarray(save_img)
        # save_img.save('sim_of/save_img_{}.png'.format(time.time()))
        # #concatenate rgb and depth_proxy
        # resize depth_proxy to 224x224

        depth_proxy = self.get_depth_proxy(obs['rgb'], obs['prev_rgb'], obs['point_mask'])

        depth_proxy = F.resize(depth_proxy, size = (224,224))
        if self.training:
            self.running_mean_var_oflow_x.update(depth_proxy[:, 0, :, :].reshape(depth_proxy.shape[0], -1))
            self.running_mean_var_oflow_y.update(depth_proxy[:, 1, :, :].reshape(depth_proxy.shape[0], -1))
        image_features = self.features_extractor(
            self._normalize_using_running_mean_std(depth_proxy, (self.running_mean_var_oflow_x,
                                                                 self.running_mean_var_oflow_y)))
        features_actor = th.cat([obs[i] for i in obs.keys() if 'critic' not in i and 'rgb' not in i
                                 and 'prev_rgb' not in i and 'point_mask' not in i], dim=1).to(th.float32)
        features_actor = th.cat([features_actor, image_features[0]], dim=1).to(th.float32)
        # features_actor = th.cat(
        #     [obs['achieved_goal'], obs['achieved_or'], obs['desired_goal'], obs['joint_angles'], obs['prev_action'],
        #      image_features[0], obs['relative_distance']], dim=1).to(th.float32)

        features = features_actor

        if self.share_features_extractor is False:
            features_critic = th.cat([obs[i] for i in obs.keys() if 'rgb' not in i
                                      and 'prev_rgb' not in i and 'point_mask' not in i],
                                     dim=1).to(th.float32)
            features_critic = th.cat([features_critic, image_features[0]], dim=1).to(th.float32)
            # features_critic = th.cat(
            #     [obs['achieved_goal'], obs['achieved_or'], obs['desired_goal'], obs['joint_angles'], obs['prev_action'],
            #         image_features[0],  obs['relative_distance'], obs['critic_pointing_cosine_sim'],
            #         obs['critic_perpendicular_cosine_sim']], dim=1).to(th.float32)
            features = (features_actor, features_critic)

        # return actor features and critic features
        #unnormalize in place
        unnormalize_recon = self._unnormalize_using_running_mean_std(image_features[1], (self.running_mean_var_oflow_x,
                                                                 self.running_mean_var_oflow_y))
        return features, depth_proxy, unnormalize_recon
    # def make_state_from_obs(self, obs):
    #     depth_features = self.extract_features(obs['depth_proxy'])
    #     #TODO: Normalize inputs
    #     robot_features = th.cat([obs['achieved_goal'], obs['desired_goal'], obs['joint_angles'], obs['prev_action']],  dim = 1)
    #     return depth_features, robot_features

    def get_distribution(
            self,
            obs: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class WHYYYY
        features, _, _ = self.extract_features(obs)
        if self.features_dim_critic_add is not None:
            actor_features = features[0]
        else:
            actor_features = features
        latent_pi, lstm_states = self._process_sequence(actor_features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(self,
            obs: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        features, _, _ = self.extract_features(obs)  # , self.vf_features_extractor)

        if self.lstm_critic is not None:
            if self.features_dim_critic_add is not None:
                critic_features = features[1]
            else:
                critic_features = features
            latent_vf, lstm_states_vf = self._process_sequence(critic_features, lstm_states, episode_starts,
                                                               self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
            self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Calculate running mean and var for observation normalization


        # Preprocess the observation if needed
        features, depth_proxy, depth_proxy_recon = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), depth_proxy, depth_proxy_recon

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # module.weight.data = module.weight.data*gain
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _predict(
            self,
            observation: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
            deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )

            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states
