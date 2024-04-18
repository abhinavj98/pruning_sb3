
import pytest
from unittest.mock import Mock, patch
import numpy as np
from ..pruning_env import PruningEnv
from .. import ROBOT_URDF_PATH, MESHES_AND_URDF_PATH
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
@pytest.fixture
def env():
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')

    env = PruningEnv(urdf_path, obj_path, label_path, renders=False, tree_count=1, make_trees=True)
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

@patch.object(PruningEnv, 'ur5')
def test_compute_reward(mock_ur5, env):
    mock_ur5.check_collisions.return_value = (False, {'collisions_acceptable': False, 'collisions_unacceptable': False})
    reward, info = env.compute_reward(np.array([1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0]), np.array([1, 1, 1, 0, 0, 0, 0]), False, None)
    assert reward == 0
    assert info == {'distance_reward': 0, 'movement_reward': 0, 'pointing_orientation_reward': 0, 'perpendicular_orientation_reward': 0, 'condition_number_reward': 0, 'termination_reward': 0, 'acceptable_collision_reward': 0, 'unacceptable_collision_reward': 0, 'slack_reward': 0, 'velocity_minimization_reward': 0}