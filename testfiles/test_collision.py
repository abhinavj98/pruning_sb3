import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_env_discrete import PruningEnv
from PPOAE.models import *
import numpy as np
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
env_kwargs = {"renders": args.RENDER, "tree_urdf_path": args.TREE_TRAIN_URDF_PATH,
              "tree_obj_path": args.TREE_TRAIN_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR}
env = PruningEnv(**env_kwargs)

# env.reset()
val = np.array([0, 0, 0, 0, 0, 0])
# Use keyboard to move the robot
while True:
    # Read keyboard input using python input
    action = get_key_pressed(env)
    # if action is wasd, then move the robot
    if ord('a') in action:
        val = np.array([1, 0, 0, 0, 0, 0])
    elif ord('d') in action:
        val = np.array([-1, 0, 0, 0, 0, 0])
    elif ord('s') in action:
        val = np.array([0, 1, 0, 0, 0, 0])
    elif ord('w') in action:
        val = np.array([0, -1, 0, 0, 0, 0])
    elif ord('q') in action:
        val = np.array([0, 0, 1, 0, 0, 0])
    elif ord('e') in action:
        val = np.array([0, 0, -1, 0, 0, 0])
    elif ord('z') in action:
        val = np.array([0, 0, 0, 1, 0, 0])
    elif ord('c') in action:
        val = np.array([0, 0, 0, -1, 0, 0])
    elif ord('x') in action:
        val = np.array([0, 0, 0, 0, 1, 0])
    elif ord('v') in action:
        val = np.array([0, 0, 0, 0, -1, 0])
    elif ord('r') in action:
        val = np.array([0, 0, 0, 0, 0, 1])
    elif ord('f') in action:
        val = np.array([0, 0, 0, 0, 0, -1])
    elif ord('t') in action:
        env.reset()
    else:
        val = np.array([0, 0, 0, 0, 0, 0])

    env.step(val)

    # env.render()
