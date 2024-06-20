import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pytest
from pruning_sb3.pruning_gym.tree import Tree
from pruning_sb3.pruning_gym import MESHES_AND_URDF_PATH
from pruning_sb3.pruning_gym.pyb_utils import pyb_utils
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
import numpy as np


@pytest.fixture
def pyb():
    pyb = pyb_utils(None, renders=False)
    return pyb


@pytest.fixture
def env():
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')
    env = PruningEnv(urdf_path, obj_path, label_path, renders=False, make_trees=True, tree_count=1)
    return env


@pytest.fixture
def tree(env, pyb):
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test', 'tree_0.urdf')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test', 'tree_0.obj')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')
    tree = Tree(env, pyb, urdf_path, obj_path, label_path, curriculum_distances=(0.4, 0.5, 0.7),
                curriculum_level_steps=(100, 200))
    return tree


def test_get_all_points():
    assert False


def test_tree_instance(env, pyb):
    urdf_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test', 'tree_0.urdf')
    obj_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test', 'tree_0.obj')
    label_path = os.path.join(MESHES_AND_URDF_PATH, 'meshes', 'trees', 'envy', 'test_labelled')
    assert Tree(env, pyb, urdf_path, label_path, obj_path)


def test_transform_obj_vertex():
    assert False


def test_is_reachable(env):
    # tree.get_reachable_points()
    tree = env.tree
    assert len(tree.reachable_points) > 0
    for point in tree.reachable_points:
        env.ur5.remove_ur5_robot()
        env.ur5.setup_ur5_arm()
        j_angles = tree.env.ur5.calculate_ik(point[0], None)
        env.ur5.set_joint_angles(j_angles)
        for _ in range(100):
            env.pyb.con.stepSimulation()
        ee_pos, _ = env.ur5.get_current_pose(tree.env.ur5.end_effector_index)
        dist = np.linalg.norm(np.array(ee_pos) - point[0], axis=-1)
        # time.sleep(10)
        assert dist < 0.05


def test_make_trees_from_folder(env):
    import glob
    glob_path = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'trees', 'envy', 'test', '*.urdf')
    # Get all the urdf files in the folder
    urdf_files = glob.glob(glob_path)

    assert len(env.trees) == len(urdf_files)


def test_make_curriculum(tree):
    # tree.get_reachable_points()
    tree.make_curriculum(None)

    for i in range(len(tree.curriculum_distances)):
        # tree.pyb_con.visualize_points(tree.curriculum_points[i])
        # import time
        # time.sleep(10)
        assert len(tree.curriculum_points[i]) != 0
