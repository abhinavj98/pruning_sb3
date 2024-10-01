import os
import sys

import numpy as np
import pytest
from pruning_sb3.pruning_gym import MESHES_AND_URDF_PATH
from pruning_sb3.pruning_gym.pruning_env import PruningEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


@pytest.fixture
def env(pos=(0, 0, 0), orient=(0, 0, 0, 1)):
    # Can you pass arguments to the fixture? Answer me:
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')

    env = PruningEnv(urdf_path, obj_path, label_path, ur5_pos=pos, ur5_or=orient, renders=False, tree_count=1,
                     make_trees=True)
    return env


def test_reset_env_variables(env):
    env.reset_env_variables()
    assert env.step_counter == 0
    assert env.sum_reward == 0
    assert env.is_goal_state == False
    assert env.collisions_acceptable == 0
    assert env.collisions_unacceptable == 0


def test_set_curriculum_level(env):
    env.set_curriculum_level(5, [5, 10, 15])
    env.set_curriculum_level(10, [5, 10, 15])
    assert env.curriculum_level == 2


def test_sample_point(env):
    distance, point = env.sample_point("PruningEnv", 1, [(0.8, (1, 2, 3))])
    assert distance == 0.8
    assert point == (1, 2, 3)


def test_is_task_done(env):
    env.step_counter = env.maxSteps + 1
    done, info = env.is_task_done()
    assert done == True
    assert info['time_limit_exceeded'] == True


def test_is_state_successful(env):
    result = env.is_state_successful(np.array([1, 1, 1]), np.array([1, 1, 1]), 0.8, 0.8)
    assert result == False


# # @patch.object(PruningEnv, 'ur5')
# def test_compute_reward(env):
#     print(env.ur5)
#     # mock_ur5.check_collisions.return_value = (False, {'collisions_acceptable': False, 'collisions_unacceptable': False})
#     reward, info = env.compute_reward(np.array([1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0]), np.array([1, 1, 1, 0, 0, 0, 0]), False, None)
#     assert reward == 0
#     assert info == {'distance_reward': 0, 'movement_reward': 0, 'pointing_orientation_reward': 0, 'perpendicular_orientation_reward': 0, 'condition_number_reward': 0, 'termination_reward': 0, 'acceptable_collision_reward': 0, 'unacceptable_collision_reward': 0, 'slack_reward': 0, 'velocity_minimization_reward': 0}

def test_set_extended_observation(env):
    # Initialize the PruningEnv class # Set a known position and orientation in the world frame
    # world_pos = np.array([1.0, 2.0, 3.0])
    # world_orient = np.array([1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2)])  # Quaternion representing no rotation
    # # init_pose_ee = env.ur5.get_current_pose(env.ur5.end_effector_index)
    # # Set these values in the environment
    # env.ur5.init_pos_base = [world_pos, world_orient]

    # Call the set_extended_observation method
    env.reset()
    env.tree_goal_pos = np.array([1, 0, 0])
    env.set_extended_observation()

    # Print the transformed position and orientation to check the results
    # print("Transformed position:", env.observation['achieved_goal'])
    assert np.isclose(env.observation['achieved_goal'], np.array([0, 0, 0]), atol=1e-2).all()
    env.robot.init_pos_base = [np.array([0, 1, 0]), np.array([0, 0, 0, 1])]
    env.set_extended_observation()
    assert False
    # print("Transformed orientation:", env.observation['achieved_or'])
