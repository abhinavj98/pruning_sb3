import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from gymnasium import spaces
import pytest
import torch as th
from src.follow_the_leader.follow_the_leader.pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import AutoEncoder
import numpy as np

from pruning_sb3.pruning_gym.running_mean_std import RunningMeanStd as rms
@pytest.fixture
def dummy_obs_space():
    observation_space = spaces.Dict({

        'desired_goal': spaces.Box(low=-5.,
                                   high=5.,
                                   shape=(3,), dtype=np.float32),
        'achieved_goal': spaces.Box(low=-5.,
                                    high=5.,
                                    shape=(3,), dtype=np.float32),
        'achieved_or': spaces.Box(low=-5.,
                                  high=5.,
                                  shape=(6,), dtype=np.float32),
        'joint_angles': spaces.Box(low=-1,
                                   high=1,
                                      shape=(6,), dtype=np.float32),
        'depth_proxy': spaces.Box(low=-1,
                                    high=1,
                                    shape=(2,224, 224), dtype=np.float32),
    })
    return observation_space
@pytest.fixture
def dummy_action_space():
    action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
    return action_space

#TODO: Constant lr not working
def test_policy_instance(dummy_obs_space, dummy_action_space):
    assert RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space, lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1)

def test_extract_features(dummy_obs_space, dummy_action_space):
    fa_kwargs = {'features_dim': 18+72, 'in_channels': 2}
    policy = RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space,\
    lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1,
    features_extractor_class=AutoEncoder, features_extractor_kwargs=fa_kwargs)
    obs = {'desired_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_or': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'joint_angles': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'depth_proxy': np.random.rand(2, 224, 224)}
    for item, value in obs.items():
        obs[item] = th.tensor(value, dtype=th.float32)
    #Make a torch tensor concatenating all the values in obs
    normalized_depth_proxy = policy._normalize_using_running_mean_std(obs['depth_proxy'], policy.running_mean_var_oflow)
    image_features = policy.features_extractor(normalized_depth_proxy)
    output = th.cat([obs['desired_goal'], obs['achieved_goal'], obs['achieved_or'], obs['joint_angles'], image_features[0]], dim=1)

    features, _ = policy.extract_features(obs)
    assert features.shape == (1, 18+72)#Autoencoder output is 72
    assert (output == features).all()

@pytest.fixture
def dummy_action_space():
    action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
    return action_space

#TODO: Constant lr not working
def test_policy_instance(dummy_obs_space, dummy_action_space):
    assert RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space, lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1)

def test_extract_features(dummy_obs_space, dummy_action_space):
    fa_kwargs = {'features_dim': 18+72, 'in_channels': 2}
    policy = RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space,\
    lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1,
    features_extractor_class=AutoEncoder, features_extractor_kwargs=fa_kwargs)
    obs = {'desired_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_or': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'joint_angles': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'depth_proxy': np.random.rand(2, 224, 224)}
    for item, value in obs.items():
        obs[item] = th.tensor(value, dtype=th.float32)
    #Make a torch tensor concatenating all the values in obs
    normalized_depth_proxy = policy._normalize_using_running_mean_std(obs['depth_proxy'], policy.running_mean_var_oflow)
    image_features = policy.features_extractor(normalized_depth_proxy)
    output = th.cat([obs['desired_goal'], obs['achieved_goal'], obs['achieved_or'], obs['joint_angles'], image_features[0]], dim=1)

    features, _ = policy.extract_features(obs)
    assert features.shape == (1, 18+72)#Autoencoder output is 72
    assert (output == features).all()

def test_extract_features_assymetric_critic(dummy_obs_space, dummy_action_space):
    fa_kwargs = {'features_dim': 18+72, 'in_channels': 2}
    policy = RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space,\
    lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1,
    features_extractor_class=AutoEncoder, features_extractor_kwargs=fa_kwargs,\
                                        share_features_extractor = False)
    obs = {'desired_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_goal': np.array([1, 2, 3]).reshape(1, -1),
           'achieved_or': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'joint_angles': np.array([1, 2, 3, 4, 5, 6]).reshape(1, -1),
           'depth_proxy': np.random.rand(2, 224, 224),
           'critic_anc': np.array([1, 2]).reshape(1, -1),
           'critic_pos': np.array([1, 2]).reshape(1, -1),}
    for item, value in obs.items():
        obs[item] = th.tensor(value, dtype=th.float32)
    #Make a torch tensor concatenating all the values in obs
    normalized_depth_proxy = policy._normalize_using_running_mean_std(obs['depth_proxy'], policy.running_mean_var_oflow)
    image_features = policy.features_extractor(normalized_depth_proxy)
    output_actor = th.cat([obs['desired_goal'], obs['achieved_goal'], obs['achieved_or'], obs['joint_angles'], image_features[0]], dim=1)
    output_critic = th.cat([obs['desired_goal'], obs['achieved_goal'], obs['achieved_or'], obs['joint_angles'], obs['critic_anc'], obs['critic_pos'], image_features[0]], dim=1)
    features = policy.extract_features(obs)
    actor_features, critic_features = features[0]
    assert actor_features.shape == (1, 18+72)#Autoencoder output is 72
    assert critic_features.shape == (1, 18+72+4)#Autoencoder output is 72
    assert (actor_features == output_actor).all()
    assert (critic_features == output_critic).all()


def test_init_weights():
    assert False

def test__normalize_using_running_mean_std(dummy_obs_space, dummy_action_space):
    policy = RecurrentActorCriticPolicy(dummy_obs_space, dummy_action_space,\
        lambda x: 0.01, lr_schedule_ae=None, lr_schedule_logstd=lambda x: 0.1)

    running_mv = rms(shape = (1, 1))
    #torch create data stream
    data_stream = th.rand(100, 1)
    running_mv.update(data_stream)
    normalized_stream = policy._normalize_using_running_mean_std(data_stream, running_mv)
    # print(normalized_stream)
    unnormalized_stream = policy._unnormalize_using_running_mean_std(normalized_stream, running_mv)
    assert th.isclose(data_stream, unnormalized_stream).all()
    assert not th.isclose(data_stream, normalized_stream).all()

