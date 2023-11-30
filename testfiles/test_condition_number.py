# (0.13449400663375854, -0.5022612810134888, 0.5729108452796936), (0.08233322092014395, 0.08148885209843372, -0.7017698639513068, 0.7029223753490557))
# ((0.13449425995349884, -0.5022624731063843, 0.5729091167449951), (0.08233936213522093, 0.08149007539678467, -0.7017685790089195, 0.7029227970202654))
# ((0.13449451327323914, -0.5022636651992798, 0.5729073882102966), (0.08234550355528829, 0.08149129879013207, -0.7017672940054546, 0.7029232186590406))
# ((0.13449475169181824, -0.5022648572921753, 0.5729056596755981)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import *
import numpy as np
import cv2
import random
import argparse
from pruning_sb3.args.args_test import args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env

def get_key_pressed(env, relevant=None):
    pressed_keys = []
    events = env.pyb.con.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:
        pressed_keys.append(key)
    return pressed_keys


# Create the ArgumentParser object
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())

parsed_args_dict = organize_args(parsed_args)
if __name__ == "__main__":
    print(parsed_args_dict['args_env']['use_optical_flow'])
    print(parsed_args_dict)
    if parsed_args_dict['args_env']['use_optical_flow'] and parsed_args_dict['args_env']['optical_flow_subproc']:
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)
    add_arg_to_env('shared_var', shared_var, ['args_train', 'args_test', 'args_record'], parsed_args_dict)

    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    env = PruningEnv(**args_test, tree_count=1)
    env.ur5.set_joint_angles((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485, -0.7777109569760938, 3.210501267258541))
    for _ in range(100):
        env.pyb.con.stepSimulation()
    # env.reset()
    val = np.array([0, 0, 0, 0, 0, 0])
    # Use keyboard to move the robot
    while True:
        # Read keyboard input using python input
        action = get_key_pressed(env)
        # if action is wasd, then move the robot
        if ord('a') in action:
            val = np.array([0.005, 0, 0, 0, 0, 0])
        elif ord('d') in action:
            val = np.array([-0.005, 0, 0, 0, 0, 0])
        elif ord('s') in action:
            val = np.array([0, 0.005, 0, 0, 0, 0])
        elif ord('w') in action:
            val = np.array([0, -0.005, 0, 0, 0, 0])
        elif ord('q') in action:
            val = np.array([0, 0, 0.005, 0, 0, 0])
        elif ord('e') in action:
            val = np.array([0, 0, -0.005, 0, 0, 0])
        elif ord('z') in action:
            val = np.array([0, 0, 0, 0.005, 0, 0])
        elif ord('c') in action:
            val = np.array([0, 0, 0, -0.005, 0, 0])
        elif ord('x') in action:
            val = np.array([0, 0, 0, 0, 0.005, 0])
        elif ord('v') in action:
            val = np.array([0, 0, 0, 0, -0.005, 0])
        elif ord('r') in action:
            val = np.array([0, 0, 0, 0, 0, 0.005])
        elif ord('f') in action:
            val = np.array([0, 0, 0, 0, 0, -0.05])
        elif ord('t') in action:
            env.reset()
        else:
            val = np.array([0, 0, 0, 0, 0, 0])
        # print(val)
        observation, reward, terminated, truncated, infos = env.step(val)
        print(env.ur5.get_joint_angles())
        # print(env.con.getLinkState(env.ur5, env.end_effector_index, 1)[6])
        # print(env.con.getLinkState(env.ur5, env.end_effector_index, 1)[7])

        # print(env.get_current_pose())
        # print(infos)
        # print(observation['desired_goal'], observation['achieved_goal'])
        # env.render()
        jacobian = env.pyb.con.calculateJacobian(env.ur5.ur5_robot, env.ur5.end_effector_index, [0, 0, 0],
                                             env.ur5.get_joint_angles(), [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        print(condition_number, 1 / condition_number)
        # # print(env.get_joint_angles())
        # print(env.target_dist)

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
