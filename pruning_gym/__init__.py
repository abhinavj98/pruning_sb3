import os

# Global URDF path pointing to robot and supports URDF

MESHES_AND_URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'meshes_and_urdf'))
ROBOT_URDF_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'ur5e', 'ur5e_cutter_new_calibrated_precise_level.urdf')
SUPPORT_AND_POST_PATH = os.path.join(MESHES_AND_URDF_PATH, 'urdf', 'supports_and_post.urdf')


label = {
    (0.117647, 0.235294, 0.039216): "SPUR",
    (0.313725, 0.313725, 0.313725): "TRUNK",
    (0.254902, 0.176471, 0.058824): "BRANCH",
    (0.235294, 0.000000, 0.000000): "WATER_BRANCH",
}
