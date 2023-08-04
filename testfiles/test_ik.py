
from gym_env_discrete import PruningEnv
from models import *
import numpy as np
import cv2
import random
import argparse
from args import args_dict

def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys 
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
env.tree.inactive()
print(env.joint_angles)
print(env.get_current_pose())
env.set_joint_angles(np.array([0,0,0,0,0,0]))

# for i in range(1000):
#     env.con.stepSimulation()
print(env.get_joint_angles())

print(env.get_current_pose())


# env.reset()
