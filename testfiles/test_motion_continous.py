from gym_env_discrete import PruningEnv
from models import *
import numpy as np
import cv2
import random
import argparse
from args import args_dict


# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)
env_kwargs = {"renders" : args.RENDER, "tree_urdf_path" :  args.TREE_TRAIN_URDF_PATH, "tree_obj_path" :  args.TREE_TRAIN_OBJ_PATH, "action_dim" : args.ACTION_DIM_ACTOR}
env = PruningEnv(**env_kwargs)


env.reset()
action = np.array([0.1,0,-0.1,0,0,0])
for i in range(100):
    action = np.array([(i-50)/100,0,0,0,0,0])
    env.step(action)
    env.render()

for i in range(100):
    action = np.array([0,(i-50)/100,0,0,0,0])
    env.step(action)
    env.render()

for i in range(100):
    action = np.array([0,0,(i-50)/100,0,0,0])
    env.step(action)
    env.render()

for i in range(100):
    action = np.array([0,0,0,(i-50)/100,0,0])
    env.step(action)
    env.render()

for i in range(100):
    action = np.array([0,0,0,0,(i-50)/100,0])
    env.step(action)
    env.render()

for i in range(100):
    action = np.array([0,0,0,0,0,(i-50)/100])
    env.step(action)
    env.render()