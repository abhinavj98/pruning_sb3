from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
import numpy as np

import torch.nn.functional as F
from torch.nn.parameter import Parameter
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:  
            self.temperature = torch.ones(1)*temperature   
        else:   
            self.temperature = Parameter(torch.ones(1))   

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1,  self.channel*2)

        return feature_keypoints

class AutoEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 128):
        super(AutoEncoder, self).__init__(observation_space, features_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding = 1), 
            nn.MaxPool2d(7),
        )
        output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        #output_conv.bias.data.fill_(0.3)
        self.fc = nn.Sequential(
            nn.Linear(32, 7*7*32),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding = 1, stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2),  # b, 16, 112, 112
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv, # b, 1, 224, 224
        )
  
    def forward(self, observation : Dict) -> th.Tensor:
        # print(observation)
        encoding = self.encoder(observation).view(-1, 32)
        fc_out = self.fc(encoding)
        recon = self.decoder(fc_out.view(-1, 32, 7, 7))
        return (encoding, recon)

class AutoEncoderSmall(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 128*7*7):
        super(AutoEncoder, self).__init__(observation_space, features_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1), 
            nn.ReLU()
        )
        output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        # output_conv.bias.data.fill_(-0.3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, padding = 1, stride=1), # 128. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 2, stride=2), # 128. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding = 1, stride=1),  # b, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2),  # b, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # b, 32, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # b, 16, 112, 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv  # b, 1, 224, 224
            #nn.ReLU()
        )

    def forward(self, observation : Dict) -> th.Tensor:
        # print(observation)
        encoding = self.encoder(observation)
        recon = self.decoder(encoding)
        return (encoding, recon)


class Actor(nn.Module):
    def __init__(self, state_dim, emb_size, action_dim, action_std):
        super(Actor, self).__init__()
        emb_ds = int(emb_size/4)
        self.output_dim = emb_size
        self.conv = nn.Sequential(
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU(),
                    )
        self.dense =  nn.Sequential(
                    nn.Linear(state_dim, emb_size),
                    nn.ReLU(),
                    #nn.Linear(emb_size, action_dim),
                    #nn.Softmax(dim=-1) #discrete action
                    )
    def forward(self, image_features, state):
        state = torch.cat((state, state, state),-1)
        dense_input = torch.cat((state, image_features),-1)
        # conv_head = self.conv(image_features)
        # if len(image_features.shape) == 4:
        #     conv_head = conv_head.view(conv_head.shape[0], -1)
        # else:
        #     conv_head = conv_head.view(1, -1)
        # dense_input = torch.cat((conv_head, state),-1) 
        action = self.dense(dense_input)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, emb_size, action_dim, action_std):
        super(Critic, self).__init__()
        self.output_dim = emb_size
        emb_ds = int(emb_size/4)
        self.conv = nn.Sequential(
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU()
                    )
        self.dense = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                #nn.Linear(emb_size, 1)
                )
    def forward(self, image_features, state):
        state = torch.cat((state, state, state),-1)
        # conv_head = self.conv(image_features)
        # if len(image_features.shape) == 4:
        #     conv_head = conv_head.view(conv_head.shape[0], -1)
        # else:
        #     conv_head = conv_head.view(1, -1)

        # dense_input = torch.cat((conv_head, state),1) 
        dense_input = torch.cat((state, image_features),-1)
        value = self.dense(dense_input)
        return value


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = Actor()
        # Value network
        self.value_net = Critic()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


# model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
# model.learn(5000)
# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 7*7*16):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = AutoEncoder(observation_space)

#         # Compute shape by doing one forward pass
#         # with th.no_grad():
#         #     n_flatten = self.cnn(
#         #         th.as_tensor(observation_space.sample()[None]).float()
#         #     ).shape[1]

#         # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.cnn(observations)#self.linear(self.cnn(observations))

# # policy_kwargs = dict(
# #     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )


class SpatialAutoEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 128):
        super(SpatialAutoEncoder, self).__init__(observation_space, features_dim)
        self.latent_space = 32
        self.output_size = 112
        self.input_size = 224
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding='same')
            # nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding = 1), 
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding = 1), 
            # nn.ReLU()
        )
        self.spatial_softmax = SpatialSoftmax(56, 56, 32)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.encoding = PositionalEncoding1D(64)
        #64*2
        output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        output_linear = nn.Linear(32*2 + 32, 7*7*16)
       # output_linear.bias.data.fill_(0.5)
        self.decoder = nn.Sequential(
            output_linear,
            nn.ReLU(),
            Reshape(-1, 16, 7, 7),
            nn.ConvTranspose2d(16, 32, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding = 1, stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # b, 16, 28,  28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2),  # b, 16, 112, 112
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
            # nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv, # b, 1, 224, 224
            #nn.ReLU()
        )
        # output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        # output_conv.bias.data.fill_(0.3)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, 3, padding = 1, stride=1), # 128. 14, 14
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 128, 2, stride=2), # 128. 14, 14
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 3, padding = 1, stride=1),  # b, 64, 28, 28
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, 2, stride=2),  # b, 64, 28, 28
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 2, stride=2),  # b, 32, 56, 56
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 2, stride=2),  # b, 16, 112, 112
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
        #     nn.ReLU(),
        #     nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
        #     nn.ReLU(),
        #     output_conv  # b, 1, 224, 224
        #     #nn.ReLU()
        # )

    def forward(self, x):
        encoding = self.encoder(x)
        argmax = self.spatial_softmax(encoding)
        maxval = self.maxpool(encoding).squeeze(-1).squeeze(-1)
        #print(features.shape)
        features = state = torch.cat((argmax, maxval),-1)
        # print((features))
        # features = self.encoding(features)
      
        recon = self.decoder(features).reshape(-1,1,self.output_size,self.output_size)
        return features,recon
