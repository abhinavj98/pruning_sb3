import os

# Global URDF path pointing to robot and supports URDF

MESHES_AND_URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'meshes_and_urdf'))
ROBOT_URDF_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'ur5e', 'cutter_field_trials.urdf')
SUPPORT_AND_POST_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'supports_and_post.urdf')


label = {
    (1.0, 0.0, 0.0): "TRUNK",
    (0.0, 1.0, 0.0): "SPUR",
    (1.0, 0.588235, 0.0): "BRANCH",
}
