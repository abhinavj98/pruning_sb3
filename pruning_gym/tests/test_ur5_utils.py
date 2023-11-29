import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pytest
from ..ur5_utils import UR5
from .. import ROBOT_URDF_PATH, MESHES_AND_URDF_PATH
from ..pyb_utils import pyb_utils

import numpy as np
@pytest.fixture
def pyb_con():
    pyb = pyb_utils(None, renders=False)
    return pyb.con

@pytest.fixture
def ur5(pyb_con):
    ur5 = UR5(pyb_con, ROBOT_URDF_PATH)
    return ur5

@pytest.fixture
def init_joint_angles():
    return (-np.pi / 2, -2., 2.16, -3.14, -1.57, np.pi)



def test_ur5_instance(pyb_con):
    assert UR5(pyb_con, ROBOT_URDF_PATH)


def test_setup_ur5_arm(ur5):
    assert ur5.ur5_robot

def test_set_collision_filter(ur5, init_joint_angles):
    #If collision filter is set, then the robot should not collide with itself
    #and the joint angles should not be similar to init
    assert np.isclose(ur5.get_joint_angles(), init_joint_angles, atol = 10e-2).all()

@pytest.mark.parametrize("joint_angles", [(-np.pi / 2, -2., 2.16, -3.14, -1.57, np.pi),
                         (-np.pi / 3, -2.4, 2.46, -1.14, -1.17, 2.7)])
def test_set_joint_angles(ur5, pyb_con, joint_angles):
    pyb_con.removeBody(ur5.ur5_robot)
    ur5.setup_ur5_arm()
    ur5.set_joint_angles(joint_angles)

    for _ in range(100):
        pyb_con.stepSimulation()
    assert np.isclose(ur5.get_joint_angles(), joint_angles, atol = 10e-2).all()

@pytest.mark.parametrize("joint_velocities", [np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                         np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),
                         np.array([+0.1, +0.1, +0.1, -0.1, -0.1, -0.1])])
def test_set_joint_velocities(ur5, pyb_con, joint_velocities):
    pyb_con.removeBody(ur5.ur5_robot)
    ur5.setup_ur5_arm()
    ur5.set_joint_velocities(joint_velocities)
    for _ in range(10):
        pyb_con.stepSimulation()
    assert np.isclose(ur5.get_joint_velocities(), joint_velocities, atol = 10e-2).all()

@pytest.mark.parametrize("end_effector_velocity", [np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                         np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),
                         np.array([+0.1, +0.1, +0.1, -0.1, -0.1, -0.1])])
def test_calculate_joint_velocities_from_end_effector_velocities(ur5, pyb_con, end_effector_velocity):
    pyb_con.removeBody(ur5.ur5_robot)
    ur5.setup_ur5_arm()
    joint_velocities, jacobian = ur5.calculate_joint_velocities_from_ee_velocity(end_effector_velocity)
    ur5.set_joint_velocities(np.array(joint_velocities))
    for _ in range(10):
        pyb_con.stepSimulation()
    actual_joint_velocities = ur5.get_joint_velocities()
    actual_end_effector_velocity = ur5.calculate_jacobian() @ actual_joint_velocities
    assert np.isclose(actual_end_effector_velocity, end_effector_velocity, atol = 10e-1).all()


@pytest.mark.parametrize("joint_angles, expected_output", [((-np.pi / 2, -2., 2.16, -3.14, -1.57, np.pi), False),
                                                     ((-2.0435414506752583, -1.961562910279876, 2.1333764856444137, -2.6531903863259485, -0.7777109569760938, 3.210501267258541), True)])
def test_check_collision(ur5, pyb_con, joint_angles, expected_output):
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test', 'tree_24.urdf')
    pyb_con.removeBody(ur5.ur5_robot)
    ur5.setup_ur5_arm()
    ur5.set_joint_angles(joint_angles)
    tree = pyb_con.loadURDF(urdf_path, [0, -0.3, 0], [0, 0, 0, 1])
    for _ in range(100):
        pyb_con.stepSimulation()
    assert ur5.check_collisions(tree, tree)[0] == expected_output

@pytest.mark.parametrize("obj_pos_diff, expected_output", [([0,-0.02, -0.01], True),
                                                        ([0,-3, -2], False)])
def test_success_collision(ur5, pyb_con, obj_pos_diff, expected_output):
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test', 'tree_24.urdf')
    pyb_con.removeBody(ur5.ur5_robot)
    ur5.setup_ur5_arm()
    pos, _ = ur5.get_current_pose(ur5.success_link_index)
    tree = pyb_con.loadURDF(urdf_path, [pos[0]+obj_pos_diff[0], pos[1]+obj_pos_diff[1], pos[2]+obj_pos_diff[2]], [0, 0, 0, 1], globalScaling=0.002)
    pyb_con.stepSimulation()
    assert ur5.check_success_collision(tree) == expected_output

# def test_get_current_pose(ur5, pyb_con):
#     joint_angles = [5.270894341,
# 3.316125579,
# 1.029744259,
# 3.473205211,
# 2.094395102,
# 1.570796327]
#     pyb_con.removeBody(ur5.ur5_robot)
#     ur5.setup_ur5_arm()
#     ur5.set_joint_angles(joint_angles)
#     for _ in range(100):
#         pyb_con.stepSimulation()
#     # import time
#     # time.sleep(50)
#     pos, orient = ur5.get_current_pose(ur5.end_effector_index)
#     pos_base, _ = ur5.get_current_pose(2)
#     assert pos == pytest.approx([-0.27667,	-0.60033, 0.51277], abs=0.1)
#     # assert orient == pytest.approx([0, 0, 0, 1], abs=0.1)

