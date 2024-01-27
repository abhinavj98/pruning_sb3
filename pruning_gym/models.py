from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch
from torch import nn
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
import numpy as np
from torchvision.transforms import functional as TVF
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


class Decoder(nn.Module):
    def __init__(self):
        output_conv = nn.Conv2d(3, 1, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1, stride=1),  # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2),  # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, padding=1, stride=1),  # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 3, padding=1, stride=1),  # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2),  # b, 16, 112, 112
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  # b, 3, 224, 224
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),  # b, 3, 224, 224
            nn.ReLU(),
            output_conv,  # b, 1, 224, 224
        )

    def forward(self, x):
        recon = self.decoder(x.view(-1, 32, 7, 7))

class AutoEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 72,  in_channels=1, size = (224, 224)):
        super(AutoEncoder, self).__init__(observation_space, features_dim)
        self.in_channels = in_channels
        self.size = size
        output_conv = nn.Conv2d(3, in_channels, 3, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding = 1), 
            nn.AvgPool2d(5, stride = 1),
        )
        output_conv = nn.Conv2d(3, in_channels, 3, padding = 1)
        #output_conv.bias.data.fill_(0.3)
        self.fc = nn.Sequential(
            nn.Linear(72, 7*7*32),
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

    def _preprocess(self, img):
        img = TVF.resize(img, size=self.size, antialias=False)
        return img

    def forward(self, image) -> th.Tensor:
        # print(observation)
        image_resized = self._preprocess(image)
        encoding = self.encoder(image_resized).view(-1, 72)
        fc_out = self.fc(encoding)
        recon = self.decoder(fc_out.view(-1, 32, 7, 7))
        return encoding, recon

class AutoEncoderSmall(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 128*7*7, size = (224, 224)):
        #Need features dim for superclass
        super(AutoEncoder, self).__init__(observation_space, features_dim)
        self.size = size
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

    def _preprocess(self, img):
        img = F.resize(img, size=self.size, antialias=False)
        return img

    def forward(self, image) -> th.Tensor:
        # print(image)
        image_resized = self._preprocess(image)
        encoding = self.encoder(image_resized)
        recon = self.decoder(encoding)
        return (encoding, recon)


class Actor(nn.Module):
    def __init__(self, state_dim, emb_size):
        super(Actor, self).__init__()
        self.output_dim = emb_size
        self.dense =  nn.Sequential(
                    nn.Linear(state_dim, emb_size*2),
                    nn.ReLU(),
                    nn.Linear(emb_size*2, emb_size),
                    nn.ReLU(),
                    )
    def forward(self, image_features, state):
        state = torch.cat((state, state),-1)
        dense_input = torch.cat((state, image_features),-1)
        action = self.dense(dense_input)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, emb_size):
        super(Critic, self).__init__()
        self.output_dim = emb_size
        self.dense = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                )
        
    def forward(self, image_features, state):
        state = torch.cat((state, state),-1)
        dense_input = torch.cat((state, image_features),-1)
        value = self.dense(dense_input)
        return value

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
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv, # b, 1, 224, 224

        )

    def forward(self, x):
        encoding = self.encoder(x)
        argmax = self.spatial_softmax(encoding)
        maxval = self.maxpool(encoding).squeeze(-1).squeeze(-1)
        features = state = torch.cat((argmax, maxval),-1)
        recon = self.decoder(features).reshape(-1,1,self.output_size,self.output_size)
        return features,recon
