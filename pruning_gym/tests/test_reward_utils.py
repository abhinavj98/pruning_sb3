from ..reward_utils import Reward
from ..pyb_utils import pyb_utils
import pytest
import numpy as np

@pytest.fixture
def reward():
    return Reward(movement_reward_scale=0.1,
                  distance_reward_scale=0.1,
                  pointing_orientation_reward_scale=0.1,
                  perpendicular_orientation_reward_scale=0.1,
                  terminate_reward_scale=0.1,
                  collision_reward_scale=0.1,
                  slack_reward_scale=0.1,
                  condition_reward_scale=0.1)
@pytest.fixture
def pyb_con():
    pyb = pyb_utils(None, renders=False)
    return pyb.con

def test_reward_instance(reward):
    assert reward

def test_calculate_perpendicular_orientation_reward():
    assert False

def test_calculate_movement_reward():
    assert False

def test_calculate_distance_reward():
    assert False

def test_calculate_pointing_orientation_reward():
    assert False

def test_calculate_condition_number_reward():
    assert False

def test_calculate_termination_reward():
    assert False

def test_calculate_acceptable_collision_reward():
    assert False

def test_calculate_unacceptable_collision_reward():
    assert False

def test_calculate_slack_reward():
    assert False

def test_calculate_velocity_minimization_reward():
    assert False

@pytest.mark.parametrize("angle, branch, expected_output", [([0, -np.pi/2, 0],[0, 0, 1],1),
                         ([0, np.pi/2, 0],[0, 0, 1], -1), ([0, 0, 0],[1, 0, 0],1),
                        ([np.pi, 0, 0],[1, 0, 0],1), ([0, 0, 0],[0, 0, 1], 0)])
def test_compute_perpendicular_cos_sim(pyb_con, angle, branch, expected_output):
    #Start is [1, 0 ,0]
    quat = pyb_con.getQuaternionFromEuler(angle)
    branch_vector = np.array(branch)
    pcs = Reward.compute_perpendicular_cos_sim(quat, branch_vector)
    assert np.isclose(pcs, expected_output, atol=10e-2)

@pytest.mark.parametrize('a, b, c, output', [([0, 1, 0],[0, 0, 1],[1, 0, 0], 0),
                                            ([0.4, 1, 0.3],[2.0,3.0, 1],[0, 2, 0], 0)])
def test_compute_perpendicular_projection(a, b, c, output):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    perp_projection = Reward.compute_perpendicular_projection(a, b, c)
    dot_product = np.dot(perp_projection, b-c)
    assert np.isclose(dot_product, output, atol=10e-2)

@pytest.mark.parametrize("current_pos, goal_pos, current_or, branch, expected_output", [([0,0,1], [0, 0, 2], [0,0,0], [0.3,0.4,0.1], 1),
                                            ([0,1,0], [0, 0, 2], [0,0,0], [1,0,0], 0.89),  ([0,1,2], [0, 0, 2], [0,0,0], [1,0,0], 0.)])
def test_compute_pointing_cos_sim(current_pos, goal_pos, current_or, branch, expected_output, pyb_con):
    quat = pyb_con.getQuaternionFromEuler(current_or)
    current_pos = np.array(current_pos)
    goal_pos = np.array(goal_pos)
    branch_vector = np.array(branch)
    pcs = Reward.compute_pointing_cos_sim(current_pos, goal_pos, quat, branch_vector)
    print(pcs)
    assert np.isclose(pcs, expected_output, atol=10e-2)
