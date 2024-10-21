import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pytest
from unittest.mock import Mock, patch
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAEWithExpert
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym import MESHES_AND_URDF_PATH
from pruning_sb3.pruning_gym.models import AutoEncoder
from gymnasium import spaces
import numpy as np
import torch as th
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import torch as th
# self.observation_space = spaces.Dict({
#             # rgb is hwc but pytorch is chw
#             'rgb': spaces.Box(low=0,
#                               high=255,
#                               shape=(self.cam_height, self.cam_width, 3),
#                               dtype=np.uint8),
#             'prev_rgb': spaces.Box(low=0,
#                                    high=255,
#                                    shape=(self.cam_height, self.cam_width, 3),
#                                    dtype=np.uint8),
#             'point_mask': spaces.Box(low=0.,
#                                      high=1.,
#                                      shape=(1, self.algo_height, self.algo_width),
#                                      dtype=np.float32),
#             'desired_goal': spaces.Box(low=-5.,
#                                        high=5.,
#                                        shape=(3,), dtype=np.float32),
#             'achieved_goal': spaces.Box(low=-5.,
#                                         high=5.,
#                                         shape=(3,), dtype=np.float32),
#             'achieved_or': spaces.Box(low=-5.,
#                                       high=5.,
#                                       shape=(6,), dtype=np.float32),
#             'joint_angles': spaces.Box(low=-1,
#                                        high=1,
#                                        shape=(12,), dtype=np.float32),
#             'prev_action_achieved': spaces.Box(low=-1., high=1.,
#                                                shape=(self.action_dim,), dtype=np.float32),
#             'relative_distance': spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32),
#             'critic_perpendicular_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),
#             'critic_pointing_cosine_sim': spaces.Box(low=-0., high=1., shape=(1,), dtype=np.float32),
#
#         })
#
# @pytest.fixture
# def dummy_action_space():
#     action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
#     return action_space
#
# @pytest.fixture
# # TODO: Constant lr not working
# def policy(dummy_obs_space, dummy_action_space):
#     return RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space, lambda x: 0.01, lr_schedule_ae=None,
#                                       lr_schedule_logstd=lambda x: 0.1)

@pytest.fixture
def env(pos=(0, 0, 0), orient=(0, 0, 0, 1)):
    # Can you pass arguments to the fixture? Answer me:
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')

    env = PruningEnv(urdf_path, obj_path, label_path, ur5_pos=pos, ur5_or=orient, renders=False, tree_count=1,
                     make_trees=True)
    return env
@pytest.fixture
def policy_kwargs():
    return {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": 128+33,
                                      "in_channels": 3,
                                      "size": (240, 424)},
        "optimizer_class": th.optim.Adam,
        "log_std_init": -0.5,
        "net_arch": dict(
            qf=[128, 64, 32],
            pi=[128, 64, 32]),
        "activation_fn": th.nn.ReLU,
        'share_features_extractor': False,
        "n_lstm_layers": 2,
        "features_dim_critic_add": 2,  # Assymetric critic
        "lstm_hidden_size": 128,
        "algo_size": (220, 424),
    }


@pytest.fixture
def algo(env, policy_kwargs):
    policy = RecurrentActorCriticPolicy
    rl_algo = RecurrentPPOAEWithExpert(path_trajectories='C:\\Users\\abhin\\PycharmProjects\\sb3bleeding\\pruning_sb3\\expert_trajectories_test', use_online_data=True, use_offline_data=True,
                                    use_ppo_offline=True, use_online_bc=True, use_awac=True, algo_size=(240,424),
                                    env = env, policy = policy, learning_rate=0.01, learning_rate_ae=0.01,
                                    learning_rate_logstd=0.01, n_steps=100, batch_size=10, n_epochs=100, ae_coeff=0.01,
                                    policy_kwargs = policy_kwargs)

    rl_algo.train_online_bc = Mock()
    rl_algo.train_awac = Mock()
    rl_algo.use_offline_data = Mock()
    rl_algo.use_online_data = Mock()
    rl_algo.use_ppo_offline = Mock()
    rl_algo.use_online_bc = Mock()
    rl_algo.collect_offline_data = Mock()
    rl_algo.collect_online_data = Mock()
    rl_algo._logger = Mock()

    #Return True whenver record is called
    rl_algo.logger.record.return_value = True
    return rl_algo

def test_algo_instance(env, policy_kwargs):
    policy = RecurrentActorCriticPolicy
    assert RecurrentPPOAEWithExpert(path_trajectories='C:\\Users\\abhin\\PycharmProjects\\sb3bleeding\\pruning_sb3\\expert_trajectories_test', use_online_data=True, use_offline_data=True,
                                    use_ppo_offline=True, use_online_bc=True, use_awac=True, algo_size=(224,224),
                                    env = env, policy = policy, learning_rate=0.01, learning_rate_ae=0.01,
                                    learning_rate_logstd=0.01, n_steps=100, batch_size=10, n_epochs=100, ae_coeff=0.01,
                                    policy_kwargs = policy_kwargs)

# def get_observation():
#     obs = {'desired_goal': th.array([1, 2, 3]).reshape(1, -1),
#            'achieved_goal': th.array([1, 2, 3]).reshape(1, -1),
#            'achieved_or': th.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
#            'joint_angles': th.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(1, -1),
#            'rgb': th.random.rand(1, 3, 240, 424),
#            'prev_rgb': th.random.rand(1, 3, 240, 424),
#            'point_mask': th.random.rand(1, 1, 240, 424),
#            'optical_flow': th.random.rand(1, 2, 240, 424),
#            'critic_perpendicular_cosine_sim': th.random.rand(1, 1),
#            'critic_pointing_cosine_sim': th.random.rand(1, 1),
#            'prev_action_achieved': th.random.rand(1, 6),
#            'relative_distance': th.random.rand(1, 3),
#            }
#     return obs

def get_observation():
    #torch tensor
    obs = {'desired_goal': th.tensor([1, 2, 3]).reshape(1, -1),
           'achieved_goal': th.tensor([1, 2, 3]).reshape(1, -1),
           'achieved_or': th.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'joint_angles': th.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(1, -1),
           'rgb': th.rand(1, 3, 240, 424),
           'prev_rgb': th.rand(1, 3, 240, 424),
           'point_mask': th.rand(1, 1, 240, 424),
           'optical_flow': th.rand(1, 2, 240, 424),
           'critic_perpendicular_cosine_sim': th.rand(1, 1),
           'critic_pointing_cosine_sim': th.rand(1, 1),
           'prev_action_achieved': th.rand(1, 6),
           'relative_distance': th.rand(1, 3),
           }
    return obs
#Make a mock generator for the rollout buffer
def fill_rollout_buffer(rl_algo):
    obs = get_observation()
    action = th.rand(1, 6)
    reward = th.rand(1, 1)
    done = th.rand(1, 1)
    episode_start = th.rand(1, 1)
    value = th.rand(1, 1)
    log_prob = th.rand(1, 1)

    single_hidden_state_shape = (1, 1, 128) # (num_layers, batch_size, hidden_size)
    device = 'cuda:0'
    lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=device),
                th.zeros(single_hidden_state_shape, device=device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=device),
                th.zeros(single_hidden_state_shape, device=device),
            ),
        )
    rl_algo.rollout_buffer.add(obs, action, reward, episode_start, value, log_prob, lstm_states=lstm_states)

def fill_expert_buffer(rl_algo):
    obs = get_observation()
    action = th.rand(1, 6)
    reward = th.rand(1, 1)
    done = th.rand(1, 1)
    episode_start = th.rand(1, 1)
    value = th.rand(1, 1)
    log_prob = th.rand(1, 1)

    single_hidden_state_shape = (1, 1, 128) # (num_layers, batch_size, hidden_size)
    device = 'cuda:0'
    lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=device),
                th.zeros(single_hidden_state_shape, device=device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=device),
                th.zeros(single_hidden_state_shape, device=device),
            ),
        )
    rl_algo.expert_buffer.add(obs, action, reward, episode_start, value, log_prob, lstm_states=lstm_states)
def test_train_offline_data(algo):
    algo.use_offline_data = True
    algo.collect_offline_data = True
    algo.use_online_data = False
    algo.collect_online_data = False
    algo.use_ppo_offline = False
    algo.use_online_bc = False
    algo.use_awac = False
    algo.train_offline_batch = Mock()
    #Set loss as a random tensor that can be called with backward
    loss = th.nn.Parameter(th.tensor([.0]))
    algo.train_offline_batch.return_value = (loss, {}, th.rand(3,240, 424), th.rand(3, 240, 424))

    for i in range(100):
        fill_expert_buffer(algo)
    #Make train expert a mock function that returns 4 values
    algo.train_expert(clip_range=0.2, clip_range_vf=0.2)
    algo.train_offline_batch.assert_called_once()

def test_train_online_data(algo):
    algo.use_offline_data = False
    algo.use_online_data = True
    algo.train_expert(clip_range=0.2, clip_range_vf=0.2)
    algo.train_online_batch.assert_called_once()

def test_train_ppo_offline(algo):
    algo.use_ppo_offline = True
    algo.train_expert(clip_range=0.2, clip_range_vf=0.2)
    algo.train_ppo_offline.assert_called_once()

def test_train_online_bc(algo):
    algo.use_online_bc = True
    algo.train_expert(clip_range=0.2, clip_range_vf=0.2)
    algo.train_online_bc.assert_called_once()

def test_train_awac(algo):
    algo.use_awac = True
    algo.train_expert(clip_range=0.2, clip_range_vf=0.2)
    algo.train_awac.assert_called_once()