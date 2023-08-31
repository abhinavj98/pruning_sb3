# (0.13449400663375854, -0.5022612810134888, 0.5729108452796936), (0.08233322092014395, 0.08148885209843372, -0.7017698639513068, 0.7029223753490557))
# ((0.13449425995349884, -0.5022624731063843, 0.5729091167449951), (0.08233936213522093, 0.08149007539678467, -0.7017685790089195, 0.7029227970202654))
# ((0.13449451327323914, -0.5022636651992798, 0.5729073882102966), (0.08234550355528829, 0.08149129879013207, -0.7017672940054546, 0.7029232186590406))
# ((0.13449475169181824, -0.5022648572921753, 0.5729056596755981)
import sys
sys.path.append("C:/Users/abhin/PycharmProjects/sb3bleeding/pruning_sb3")
from gym_env_discrete import PruningEnv
from PPOAE.models import *
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
env = PruningEnv(**env_kwargs, tree_count=1)


# env.reset()
val = np.array([0,0,0,0,0,0])
#Use keyboard to move the robot
while True:
    #Read keyboard input using python input
    action = get_key_pressed(env)
    #if action is wasd, then move the robot
    if ord('a') in action:
        val = np.array([0.05,0,0,0,0,0])
    elif ord('d') in action:
        val = np.array([-0.05,0,0,0,0,0])
    elif ord('s') in action:
        val = np.array([0,0.05,0,0,0,0])
    elif ord('w') in action:
        val = np.array([0,-0.05,0,0,0,0])
    elif ord('q') in action:
        val = np.array([0,0,0.05,0,0,0])
    elif ord('e') in action:
        val = np.array([0,0,-0.05,0,0,0])
    elif ord('z') in action:
        val = np.array([0,0,0,0.05,0,0])
    elif ord('c') in action:
        val = np.array([0,0,0,-0.05,0,0])
    elif ord('x') in action:
        val = np.array([0,0,0,0,0.05,0])
    elif ord('v') in action:
        val = np.array([0,0,0,0,-0.05,0])
    elif ord('r') in action:
        val = np.array([0,0,0,0,0,0.05])
    elif ord('f') in action:
        val = np.array([0,0,0,0,0,-0.05])
    elif ord('t') in action:
        env.reset()
    else:
        val = np.array([0,0,0,0,0,0])
 
    observation, reward, terminated, truncated, infos = env.step(val)
    # print(env.get_current_pose())
    # print(infos)
    # print(observation['desired_goal'], observation['achieved_goal'])
    # env.render()
    # jacobian = env.con.calculateJacobian(env.ur5, env.end_effector_index, [0,0,0], env.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
    # jacobian = np.vstack(jacobian)
    # condition_number = np.linalg.cond(jacobian)
    # print(condition_number, 1/condition_number)
    # print(env.get_joint_angles())
    print(env.target_dist)
    
        
"""
print("Initial position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
try:
    action = int(input('action please'))
except:
    continue
if action == 0:
    quit()
if action > 12:
    print("Wrong action")
    continue
print(env.rev_actions[action])

r = env.step(action, False)
print(r[1][-1])
print("Final position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
"""