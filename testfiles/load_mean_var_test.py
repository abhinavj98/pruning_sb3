import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.running_mean_std import RunningMeanStd
import pickle
with open("C:/Users/abhin/PycharmProjects/sb3bleeding/logs/test/mean_std.obj", "rb") as f:
    mean_std = pickle.load(f)
print(mean_std)
print(mean_std.mean)
print(mean_std.var)