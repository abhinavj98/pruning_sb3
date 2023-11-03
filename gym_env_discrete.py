from typing import Optional, Tuple, Any, List
import glob
import math
import os
import pickle
import random
from collections import namedtuple

import gymnasium as gym
import numpy as np
import pybullet
import pywavefront
from gymnasium import spaces
from nptyping import NDArray, Shape, Float
from pybullet_utils import bullet_client as bc

from optical_flow import OpticalFlow

# Global URDF path pointing to robot and supports URDF
ROBOT_URDF_PATH = "meshes_and_urdf/urdf/ur5e/ur5e_cutter_new_calibrated_precise_level.urdf"  # "meshes_and_urdf/urdf
# /ur5e_with_camera.urdf"#
SUPPORT_AND_POST_PATH = "meshes_and_urdf/urdf/supports_and_post.urdf"


class PruningEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    # optical_flow_model = OpticalFlow()

    def __init__(self, tree_urdf_path: str, tree_obj_path: str, renders: bool = False, max_steps: int = 500,
                 learning_param: float = 0.05, tree_count: int = 9999, width: int = 224, height: int = 224,
                 evaluate: bool = False, num_points: Optional[int] = None, action_dim: int = 12,
                 name: str = "PruningEnv",
                 terminate_on_singularity: bool = True, action_scale: int = 1, movement_reward_scale: int = 1,
                 distance_reward_scale: int = 1, condition_reward_scale: int = 1, terminate_reward_scale: int = 1,
                 collision_reward_scale: int = 1, slack_reward_scale: int = 1,
                 perpendicular_orientation_reward_scale: int = 1, pointing_orientation_reward_scale: int = 1,
                 use_optical_flow: bool = False, optical_flow_subproc: bool = False,
                 shared_var: Tuple[Optional[Any], Optional[Any]] = (None, None), scale: bool = False) -> None:
        super(PruningEnv, self).__init__()

        assert tree_urdf_path is not None
        assert tree_obj_path is not None

        # TODO: Assumes shared variables is always present, to not use shared var have to pass (None, None). Make
        #  shared_var as None
        self.shared_queue = shared_var[0]
        self.shared_dict = shared_var[1]
        self.pid = os.getpid()
        # self.shared_dict[self.pid] = None
        # Pybullet GUI variables
        self.renders = renders
        self.render_mode = 'rgb_array'
        self.eval = evaluate
        # Obj/URDF paths
        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        # Gym variables
        self.name = name
        self.action_dim = action_dim
        self.stepCounter = 0
        self.maxSteps = max_steps
        self.tree_count = tree_count
        self.action_scale = action_scale
        self.terminated = False
        self.terminate_on_singularity = terminate_on_singularity
        self.use_optical_flow = use_optical_flow
        self.optical_flow_subproc = optical_flow_subproc
        if self.use_optical_flow:
            if not self.optical_flow_subproc:
                self.optical_flow_model = OpticalFlow(subprocess=False)
        #     self.optical_flow_model = OpticalFlow()
        # Reward variables

        self.movement_reward_scale = movement_reward_scale
        self.distance_reward_scale = distance_reward_scale
        self.condition_reward_scale = condition_reward_scale
        self.terminate_reward_scale = terminate_reward_scale
        self.collision_reward_scale = collision_reward_scale
        self.slack_reward_scale = slack_reward_scale
        self.perpendicular_orientation_reward_scale = perpendicular_orientation_reward_scale
        self.pointing_orientation_reward_scale = pointing_orientation_reward_scale
        self.sum_reward = 0.0
        self.scale = scale
        # Set up pybullet
        if self.renders:
            self.con = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.con = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.con.setTimeStep(5. / 240.)
        self.con.setGravity(0, 0, -10)
        self.con.setRealTimeSimulation(False)

        self.con.resetDebugVisualizerCamera(cameraDistance=1.06, cameraYaw=-120.3, cameraPitch=-12.48,
                                            cameraTargetPosition=[-0.3, -0.06, 0.4])

        self.observation_space = spaces.Dict({
            'depth': spaces.Box(low=-1.,
                                high=1.0,
                                shape=((2, 224, 224) if self.use_optical_flow else (1, 224, 224)), dtype=np.float32),
            'desired_goal': spaces.Box(low=-5.,
                                       high=5.,
                                       shape=(3,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-5.,
                                        high=5.,
                                        shape=(3,), dtype=np.float32),
            'achieved_or': spaces.Box(low=-5.,
                                        high=5.,
                                        shape=(6,), dtype=np.float32),
            'joint_angles': spaces.Box(low=-1,
                                       high=1,
                                       shape=(12,), dtype=np.float32),
            'joint_velocities': spaces.Box(low=-6,
                                           high=6,
                                           shape=(6,), dtype=np.float32),
            'prev_action': spaces.Box(low=-1., high=1.,
                                      shape=(self.action_dim,), dtype=np.float32),
            # 'cosine_sim': spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32),
            'close_to_goal': spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            'relative_distance': spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_dim,), dtype=np.float32)

        self.prev_action = np.zeros(self.action_dim)
        self.action = np.zeros(self.action_dim)
        # self.grayscale = np.zeros((224, 224))
        self.rgb = np.zeros((224, 224, 3))
        self.prev_rgb = np.zeros((224, 224, 3))

        self.cosine_sim = 0.5
        self.reset_counter = 0
        self.randomize_tree_count = 1
        self.sphereUid = -1
        self.learning_param = learning_param

        # setup robot arm:
        self.setup_ur5_arm()
        self.reset_env_variables()
        # Camera parameters
        self.near_val = 0.01
        self.far_val = 3
        self.height = height
        self.width = width
        # TODO: Correct camera parameters?
        self.proj_mat = self.con.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=self.near_val,
            farVal=self.far_val)

        # Tree parameters
        self.tree_goal_pos = np.array([1, 0, 0])  # initial object pos
        self.tree_goal_branch = np.array([0, 0, 0])
        pos = None
        scale = None
        if "envy" in self.tree_urdf_path:
            pos = np.array([0, -0.7, 0])
            scale = 1
        elif "ufo" in self.tree_urdf_path:
            pos = np.array([-0.5, -0.8, -0.3])
            scale = 1

        assert scale is not None
        assert pos is not None
        self.trees = Tree.make_trees_from_folder(self, self.tree_urdf_path, self.tree_obj_path, pos=pos,
                                                 orientation=np.array([0, 0, 0, 1]), scale=scale, num_points=num_points,
                                                 num_trees=self.tree_count)
        self.tree = random.sample(self.trees, 1)[0]
        self.supports = -1
        self.tree.active()
        self.create_background()

        # Debug parameters
        self.debug_line = -1
        self.debug_line_1 = -1
        self.debug_cur_point = -1
        self.debug_des_point = -1
        self.debug_cur_perp = -1
        self.debug_des_perp = -1
        self.debug_branch = -1

        self.eval_counter = 0
    def reset_env_variables(self) -> None:
        # Env variables that will change
        self.observation: dict = dict()
        self.stepCounter = 0
        self.sum_reward = 0
        self.terminated = False
        self.singularity_terminated = False
        self.wrong_success = False
        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0
        self.target_dist = 1.

    def setup_ur5_arm(self) -> None:

        self.camera_link_index = 11
        self.end_effector_index = 13
        self.success_link_index = 14
        flags = self.con.URDF_USE_SELF_COLLISION
        self.ur5 = self.con.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.], [0, 0, 0, 1], flags=flags)

        self.num_joints = self.con.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                               "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo",  # type: ignore
                                     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                      "controllable"])

        self.joints = dict()
        for i in range(self.num_joints):
            info = self.con.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            # print("Joint Name: ", jointName, "Joint ID: ", jointID)
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)  # type: ignore
            print(jointID, jointName)
            if info.type == "REVOLUTE":
                self.con.setJointMotorControl2(self.ur5, info.id, self.con.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info

        # TO SET CUTTER DISABLE COLLISIONS WITH SELF
        self.con.setCollisionFilterPair(self.ur5, self.ur5, 9, 11, 0)
        self.con.setCollisionFilterPair(self.ur5, self.ur5, 8, 11, 0)
        self.con.setCollisionFilterPair(self.ur5, self.ur5, 10, 11, 0)
        self.con.setCollisionFilterPair(self.ur5, self.ur5, 7, 11, 0)
        self.con.setCollisionFilterPair(self.ur5, self.ur5, 6, 11, 0)

        self.init_joint_angles = (-np.pi / 2, -2., 2.16, -3.14, -1.57, np.pi)
        self.set_joint_angles(self.init_joint_angles)
        for i in range(1000):
            self.con.stepSimulation()

        self.init_pos = self.con.getLinkState(self.ur5, self.end_effector_index)
        self.joint_velocities = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        self.previous_pose = np.array([0, 0, 0, 0, 0, 0, 0]).astype(np.float32)

    def create_background(self) -> None:
        # wall_folder = os.path.join('models', 'wall_textures')
        #TODO: Replace all static paths with os.path.join
        if os.name == "posix":
            self.wall_texture = self.con.loadTexture(
                "meshes_and_urdf/textures/bark_willow_02/bark_willow_02_diff_4k.jpg")
        else:
            self.wall_texture = self.con.loadTexture(
                "C:/Users/abhin/PycharmProjects/sb3bleeding/pruning_sb3/meshes_and_urdf/textures/leaves-dead.png")
        self.floor_dim = [0.01, 5, 5]
        floor_viz = self.con.createVisualShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                               )
        floor_col = self.con.createCollisionShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                                  )

        self.floor_id = self.con.createMultiBody(baseMass=0, baseVisualShapeIndex=floor_viz,
                                                 baseCollisionShapeIndex=floor_col, basePosition=[0, 0, 0],
                                                 baseOrientation=list(
                                                     self.con.getQuaternionFromEuler([0, np.pi / 2, 0])))

        wall_viz = self.con.createVisualShape(shapeType=self.con.GEOM_BOX, halfExtents=[0.01, 5, 5],
                                              )
        wall_col = self.con.createCollisionShape(shapeType=self.con.GEOM_BOX, halfExtents=[0.01, 5, 5],
                                                 )
        self.wall_id = self.con.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_viz,
                                                baseCollisionShapeIndex=wall_col, basePosition=[0, -2, 5],
                                                baseOrientation=list(
                                                    self.con.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])))

        side_wall_viz_1 = self.con.createVisualShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                                     )
        side_wall_col_1 = self.con.createCollisionShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                                        )

        self.side_wall_1_id = self.con.createMultiBody(baseMass=0, baseVisualShapeIndex=side_wall_viz_1,
                                                       baseCollisionShapeIndex=side_wall_col_1, basePosition=[-5, 0, 5],
                                                       baseOrientation=list(
                                                           self.con.getQuaternionFromEuler([0, 0, 0])))
        side_wall_viz_2 = self.con.createVisualShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                                     )
        side_wall_col_2 = self.con.createCollisionShape(shapeType=self.con.GEOM_BOX, halfExtents=self.floor_dim,
                                                        )
        self.side_wall_2_id = self.con.createMultiBody(baseMass=0, baseVisualShapeIndex=side_wall_viz_2,
                                                       baseCollisionShapeIndex=side_wall_col_2, basePosition=[5, 0, 5],
                                                       baseOrientation=list(
                                                           self.con.getQuaternionFromEuler([0, 0, 0])))
        self.con.changeVisualShape(objectUniqueId=self.side_wall_1_id, linkIndex=-1,
                                   textureUniqueId=self.wall_texture)
        self.con.changeVisualShape(objectUniqueId=self.side_wall_2_id, linkIndex=-1,
                                   textureUniqueId=self.wall_texture)

        self.con.changeVisualShape(objectUniqueId=self.floor_id, linkIndex=-1,
                                   textureUniqueId=self.wall_texture)
        self.con.changeVisualShape(objectUniqueId=self.wall_id, linkIndex=-1,
                                   textureUniqueId=self.wall_texture)

        # self.con.changeVisualShape(objectUniqueId=self.tree.tree_urdf, linkIndex=-1,
        #                            textureUniqueId=self.wall_texture)

    def set_joint_angles(self, joint_angles: Tuple[float, float, float, float, float, float]) -> None:
        """Set joint angles using pybullet motor control"""
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        self.con.setJointMotorControlArray(
            self.ur5, indexes,
            self.con.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces
        )

    from nptyping import NDArray, Shape, Float
    def set_joint_velocities(self, joint_velocities: NDArray[Shape['6, 1'], Float]) -> None:
        """Set joint velocities using pybullet motor control"""
        velocities = []
        indexes = []
        forces = []
        singularity = False
        if (abs(joint_velocities) > 0.25).any():
            singularity = True
            joint_velocities = np.clip(joint_velocities,-np.pi/4, np.pi/4)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
        # print(joint_velocities)
        # if (joint_velocities > 1).any():
        #     input()
        maxForce = 500
        self.con.setJointMotorControlArray(self.ur5,
                                           indexes,
                                           controlMode=self.con.VELOCITY_CONTROL,
                                           targetVelocities=joint_velocities,
                                           )


        return singularity

    def calculate_joint_velocities_from_end_effector_velocity(self,
                                                              end_effector_velocity: NDArray[Shape['6, 1'], Float]) -> \
            NDArray[Shape['6, 1'], Float]:
        """Calculate joint velocities from end effector velocity using jacobian"""
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0, 0, 0], self.get_joint_angles(),
                                              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        jacobian = np.vstack(jacobian)
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
        return joint_velocities

    def get_joint_angles(self) -> Tuple[float, float, float, float, float, float]:
        """Return joint angles"""
        j = self.con.getJointStates(self.ur5, [3, 4, 5, 6, 7, 8])
        joints = tuple((i[0] for i in j))
        return joints  # type: ignore

    def check_collisions(self) -> Tuple[bool, dict]:
        """Check if there are any collisions between the robot and the environment
        Returns: Dictionary with information about collisions (Acceptable and Unacceptable)
        """
        collisions_acceptable = self.con.getContactPoints(bodyA=self.ur5, bodyB=self.tree.tree_urdf)
        collisions_unacceptable = self.con.getContactPoints(bodyA=self.ur5, bodyB=self.tree.supports)
        collision_info = {"collisions_acceptable": False, "collisions_unacceptable": False}
        for i in range(len(collisions_unacceptable)):
            # print("collision")
            if collisions_unacceptable[i][-6] < 0:
                collision_info["collisions_unacceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True, collision_info

        for i in range(len(collisions_acceptable)):
            # print("collision")
            if collisions_acceptable[i][-6] < 0:
                collision_info["collisions_acceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True, collision_info
        return False, collision_info
    def check_success_collision(self) -> bool:
        """Check if there are any collisions between the robot and the environment
        Returns: Dictionary with information about collisions (Acceptable and Unacceptable)
        """
        collisions_success = self.con.getContactPoints(bodyA=self.ur5, bodyB=self.tree.tree_urdf, linkIndexA=self.success_link_index)
        for i in range(len(collisions_success)):
            # print("collision")
            if collisions_success[i][-6] < 0:
                # collision_info["collisions_unacceptable"] = True
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True
        return False

    def calculate_ik(self, position: Tuple[float, float, float], orientation: Optional[Tuple[float, float, float, float]]) -> \
            Tuple[float, float, float, float, float, float]:
        """Calculates joint angles from end effector position and orientation using inverse kinematics"""
        lower_limits = [-math.pi] * 6
        upper_limits = [math.pi] * 6
        joint_ranges = [2 * math.pi] * 6

        joint_angles = self.con.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, orientation,
            jointDamping=[0.01] * 6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges#, restPoses=self.init_joint_angles
        )
        return joint_angles

    def get_current_pose(self, index: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Returns current pose of the end effector. Pos wrt end effector, orientation wrt world"""
        linkstate = self.con.getLinkState(self.ur5, index, computeForwardKinematics=True)
        position, orientation = linkstate[4], linkstate[1]  # Position wrt end effector, orientation wrt COM
        return position, orientation

    # TODO: Better types for getCameraImage
    def get_image_at_curr_pose(self) -> List:
        """Take the current pose of the end effector and set the camera to that pose"""
        CAMERA_BASE_OFFSET = np.array([-0.063179, 0.077119, 0.0420027])
        pose, orientation = self.get_current_pose(self.camera_link_index)
        # CAMERA_BASE_OFFSET = np.array([0.01, 0.005, 0.01 ])  # TODO: Change camera position
        pose = pose
        tilt = -np.pi / 18
        tilt_rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        tf = np.identity(4)
        tf[:3, 3] = CAMERA_BASE_OFFSET
        rot_tf = np.identity(4)
        rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        rot_tf[:3, :3] = rot_mat
        rot_tf[:3, 3] = pose
        # cam_pos, cam_orient = self.con.multiplyTransforms(pose, [0,0,1,1], pose + CAMERA_BASE_OFFSET, orientation)
        tf = rot_tf@tf
        # Initial vectors
        init_camera_vector = np.array([0, 0, 1])@tilt_rot  #
        init_up_vector = np.array([0, 1, 0])@tilt_rot  #
        # Rotated vectors
        camera_vector = rot_mat.dot(init_camera_vector)
        up_vector = rot_mat.dot(init_up_vector)
        view_matrix = self.con.computeViewMatrix(tf[:3,3], tf[:3,3] + 0.1 * camera_vector, up_vector)

        return self.con.getCameraImage(self.width, self.height, viewMatrix=view_matrix, projectionMatrix=self.proj_mat,
                                       renderer=self.con.ER_BULLET_HARDWARE_OPENGL,
                                       flags=self.con.ER_NO_SEGMENTATION_MASK, lightDirection=[1, 1, 1])

    @staticmethod
    def seperate_rgbd_rgb_d(rgbd: List, h: int = 224, w: int = 224) -> Tuple[NDArray, NDArray]:
        """Seperate rgb and depth from the rgbd image, return RGB and depth"""
        rgb = np.array(rgbd[2]).reshape(h, w, 4) / 255
        rgb = rgb[:, :, :3]
        depth = np.array(rgbd[3]).reshape(h, w)
        return rgb, depth

    @staticmethod
    def linearize_depth(depth: NDArray, far_val: float, near_val: float):
        """OpenGL returns contracted depth, linearize it"""
        depth_linearized = near_val / (far_val - (far_val - near_val) * depth + 0.00000001)
        return depth_linearized

    def get_rgbd_at_cur_pose(self) -> Tuple[NDArray, NDArray]:
        """Get RGBD image at current pose"""
        # cur_p = self.get_current_pose(self.camera_link_index)
        rgbd = self.get_image_at_curr_pose()
        rgb, depth = self.seperate_rgbd_rgb_d(rgbd)
        depth = depth.astype(np.float32)
        depth = self.linearize_depth(depth, self.far_val, self.near_val) - 0.5
        return rgb, depth

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[dict, dict]:
        """Environment reset function"""
        super().reset(seed=seed)
        random.seed(seed)
        self.reset_env_variables()
        self.reset_counter += 1
        # Remove and add tree to avoid collisions with tree while resetting
        self.tree.inactive()
        self.con.removeBody(self.ur5)
        # Remove debug items
        self.con.removeUserDebugItem(self.debug_branch)
        self.con.removeUserDebugItem(self.debug_line)
        self.con.removeUserDebugItem(self.debug_line_1)
        self.con.removeUserDebugItem(self.debug_cur_point)
        self.con.removeUserDebugItem(self.debug_des_point)
        self.con.removeUserDebugItem(self.debug_des_perp)
        self.con.removeUserDebugItem(self.debug_cur_perp)
        # Sample new tree if reset_counter is a multiple of randomize_tree_count
        if self.reset_counter % self.randomize_tree_count == 0:
            self.tree = random.sample(self.trees, 1)[0]

        # Create new ur5 arm body
        self.setup_ur5_arm()

        # Sample new point
        if "eval" in self.name:
            random_point = self.tree.reachable_points[self.eval_counter%len(self.tree.reachable_points)]
            self.eval_counter += 1
            print("Eval counter: ", self.eval_counter, "Point: ", random_point)

        else:
            random_point = random.sample(self.tree.reachable_points, 1)[0]
        if "record" in self.name:
            """Display red sphere during evaluation"""
            self.con.removeBody(self.sphereUid)
            colSphereId = -1
            visualShapeId = self.con.createVisualShape(self.con.GEOM_SPHERE, radius=0.005, rgbaColor=[1, 0, 0, 1])
            self.sphereUid = self.con.createMultiBody(0.0, colSphereId, visualShapeId,
                                                      [random_point[0][0], random_point[0][1], random_point[0][2]],
                                                      [0, 0, 0, 1])
        # self.set_joint_angles(self.init_joint_angles)
        self.set_joint_angles(
            self.calculate_ik((random_point[0][0], self.init_pos[0][1], random_point[0][2]), self.init_pos[1]))
        for i in range(500):
            self.con.stepSimulation()
        self.tree_goal_pos = random_point[0]
        self.tree_goal_branch = random_point[1]
        self.tree.active()
        curr_pose = self.get_current_pose(self.end_effector_index)
        self.init_distance = np.linalg.norm(self.tree_goal_pos - self.init_pos[0]) + 1e-4
        self.init_point_cosine_sim = self.compute_pointing_cos_sim(curr_pose[0], self.tree_goal_pos, curr_pose[1], self.tree_goal_branch) + 1e-4
        self.init_perp_cosine_sim = self.compute_perpendicular_cos_sim(curr_pose[1], self.tree_goal_branch) + 1e-4


        # Add debug branch
        self.debug_branch = self.con.addUserDebugLine(self.tree_goal_pos - 50 * self.tree_goal_branch,
                                                      self.tree_goal_pos + 50 * self.tree_goal_branch, [1, 0, 0], 200)

        self.set_extended_observation()
        info = dict()  # type: ignore
        # Make info analogous to one in step function
        return self.observation, info

    def step(self, action: NDArray[Shape['6, 1'], Float]) -> Tuple[dict, float, bool, bool, dict]:
        # remove debug line

        previous_pose = self.get_current_pose(self.end_effector_index)
        # convert two tuples into one array

        self.previous_pose = np.hstack((previous_pose[0], previous_pose[1]))
        self.prev_joint_velocities = self.joint_velocities
        self.prev_rgb = self.rgb
        self.con.removeUserDebugItem(self.debug_line)

        self.con.removeUserDebugItem(self.debug_line_1)
        self.action = action

        action = action * self.action_scale
        self.joint_velocities = self.calculate_joint_velocities_from_end_effector_velocity(action)
        singularity = self.set_joint_velocities(self.joint_velocities)

        for i in range(5):
            self.con.stepSimulation()
            # print(self.con.getJointStateMultiDof(self.ur5, self.end_effector_index))
            # if self.renders: time.sleep(5./240.) 

        # Next 2 lines keep in the same order, need observations before reward
        self.set_extended_observation()
        reward, reward_infos = self.compute_reward(self.desired_pos, np.hstack((self.achieved_pos, self.achieved_or)),
                                                   self.previous_pose, singularity,
                                                   None)

        self.sum_reward += reward
        # self.debug_line = self.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0, 0, 1], 20)
        self.stepCounter += 1

        done, terminate_info = self.is_task_done()
        truncated = terminate_info['time_limit_exceeded']
        terminated = terminate_info['goal_achieved'] or terminate_info['singularity_achieved']

        infos = {'is_success': False, "TimeLimit.truncated": False}  # type: ignore
        if False:#terminate_info['time_limit_exceeded']:
            infos["TimeLimit.truncated"] = True  # type: ignore
            infos["terminal_observation"] = self.observation  # type: ignore

        if self.terminated is True:
            infos['is_success'] = True

        if truncated or terminated:
            infos['episode'] = {"l": self.stepCounter, "r": self.sum_reward}  # type: ignore
            print("Episode Length: ", self.stepCounter)
            infos["pointing_cosine_sim_error"] = self.orientation_point_value
            infos["perpendicular_cosine_sim_error"] = self.orientation_perp_value
            infos["euclidean_error"] = self.target_dist

        # infos['episode'] = {"l": self.stepCounter,  "r": reward}
        infos.update(reward_infos)
        # return self.observation, reward, done, infos
        # v26
        return self.observation, reward, terminated, truncated, infos

    def render(self, mode="rgb_array") -> NDArray:  # type: ignore
        size = [512, 512]
        view_matrix = self.con.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.3, -0.06, 1.3], distance=1.06,
                                                                 yaw=-120.3, pitch=-12.48, roll=0, upAxisIndex=2)
        proj_matrix = self.con.computeProjectionMatrixFOV(fov=60, aspect=float(size[0]) / size[1], nearVal=0.1,
                                                          farVal=100.0)
        img_rgbd = self.con.getCameraImage(size[0], size[1], view_matrix, proj_matrix,
                                           renderer=self.con.ER_BULLET_HARDWARE_OPENGL,
                                           flags=self.con.ER_NO_SEGMENTATION_MASK)
        # , renderer = self.con.ER_BULLET_HARDWARE_OPENGL)
        img_rgb, _ = self.seperate_rgbd_rgb_d(img_rgbd, size[1], size[0])
        if mode == "human":
            import cv2
            cv2.imshow("img", (img_rgb * 255).astype(np.uint8))
            cv2.waitKey(1)

        return img_rgb

    def close(self) -> None:
        self.con.disconnect()

    def set_extended_observation(self) -> None:
        """
        The observations are the current position, the goal position, the current orientation, the current depth
        image, the current joint angles and the current joint velocities
        """
        self.prev_action = self.action

        tool_pos, tool_orient = self.get_current_pose(self.end_effector_index)
        self.achieved_pos = np.array(tool_pos).astype(np.float32)
        self.achieved_or = np.array(tool_orient).astype(np.float32)

        self.desired_pos = np.array(self.tree_goal_pos).astype(np.float32)

        self.rgb, self.depth = self.get_rgbd_at_cur_pose()
        # self.grayscale = self.rgb.mean(axis=2).astype(np.uint8)

        self.joint_angles = np.array(self.get_joint_angles()).astype(np.float32)

        init_pos = np.array(self.init_pos[0]).astype(np.float32)
        init_or = np.array(self.init_pos[1]).astype(np.float32)

        self.observation['achieved_goal'] = self.achieved_pos
        #Convert orientation into 6D form for continuity
        achieved_or_mat = np.array(self.con.getMatrixFromQuaternion(self.achieved_or)).reshape(3, 3)
        # print(achieved_or_mat[:, :])
        achieved_or_6d = achieved_or_mat[:, :2].reshape(6,).astype(np.float32)
        self.observation['achieved_or'] = achieved_or_6d
        # , np.array(
        #     self.con.getEulerFromQuaternion(self.achieved_or)) - np.array(self.con.getEulerFromQuaternion(init_or))))
        self.observation['desired_goal'] = self.desired_pos

        # print(np.array(self.con.getEulerFromQuaternion(self.achieved_or)) - np.array(self.con.getEulerFromQuaternion(init_or)))
        if self.use_optical_flow:
            # #if subprocvenv add the rgb to the queue and wait for the optical flow to be calculated
            # if self.subprocvenv:
            #     self.rgb_queue.put(self.rgb)
            #     self.observation['depth'] = self.optical_flow_queue.get()
            if self.optical_flow_subproc:
                self.shared_queue.put((self.rgb, self.prev_rgb, self.pid))
                while not self.pid in self.shared_dict.keys():
                    pass
                optical_flow = self.shared_dict[self.pid]
                self.observation['depth'] = optical_flow
                # ((optical_flow - optical_flow.min()) / (
                # optical_flow.max() - optical_flow.min() + 1e-6))
                # self.shared_dict[self.pid] = None
                del self.shared_dict[self.pid]
            else:
                optical_flow = self.optical_flow_model.calculate_optical_flow(self.rgb, self.prev_rgb)
                self.observation['depth'] = optical_flow
            # self.observation['depth'] = self.optical_flow_model.calculate_optical_flow(self.rgb,
            #  self.prev_rgb)  # TODO: rename depth
        else:
            self.observation['depth'] = np.expand_dims(self.depth.astype(np.float32), axis=0)

        # self.observation['cosine_sim'] = np.array(self.cosine_sim).astype(np.float32).reshape(
        #     1, )  # DO NOT PASS THIS AS STATE - JUST HERE FOR COSINE SIM PREDICTOR
        #Convert joint angles to sin and cos
        encoded_joint_angles = np.hstack((np.sin(self.joint_angles), np.cos(self.joint_angles)))
        self.observation['joint_angles'] = encoded_joint_angles

        self.observation['joint_velocities'] = self.joint_velocities
        #Action actually achieved
        achieved_vel = np.array(self.con.getLinkState(self.ur5, self.end_effector_index, 1)[6])
        achieved_ang_vel = np.array(self.con.getLinkState(self.ur5, self.end_effector_index, 1)[7])
        self.observation['prev_action'] = np.hstack((achieved_vel, achieved_ang_vel))
        if self.target_dist < self.learning_param:
            self.observation['close_to_goal'] = np.array(1).astype(np.float32).reshape(1, )
        else:
            self.observation['close_to_goal'] = np.array(0).astype(np.float32).reshape(1, )
        self.observation['relative_distance'] = self.achieved_pos - self.desired_pos

    def is_task_done(self) -> Tuple[bool, dict]:
        # NOTE: need to call compute_reward before this to check termination!
        time_limit_exceeded = self.stepCounter >= self.maxSteps
        singularity_achieved = self.singularity_terminated
        wrong_success = self.wrong_success
        goal_achieved = self.terminated
        c = (
                self.wrong_success is True or self.singularity_terminated is True or self.terminated is True or self.stepCounter > self.maxSteps)
        terminate_info = {"time_limit_exceeded": time_limit_exceeded, "singularity_achieved": singularity_achieved,
                          "wrong_success": wrong_success,
                          "goal_achieved": goal_achieved}
        return c, terminate_info

    def get_condition_number(self) -> float:
        # get jacobian
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0, 0, 0], self.get_joint_angles(),
                                              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        return condition_number

    def compute_pointing_orientation_reward(self, achieved_pos: NDArray[Shape['3, 1'], Float],
                                            desired_pos: NDArray[Shape['3, 1'], Float],
                                            achieved_or: NDArray[Shape['4, 1'], Float],
                                            previous_pos: NDArray[Shape['3, 1'], Float],
                                            previous_or: NDArray[Shape['4, 1'], Float],
                                            branch_vector: NDArray[Shape['3, 1'], Float]) -> Tuple[float, float]:
        # Orientation reward is computed as the dot product between the current orientation and the perpendicular vector to the end effector and goal pos vector
        # This is to encourage the end effector to be perpendicular to the branch

        # Perpendicular vector to branch vector
        # perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector + desired_pos)
        # perpendicular_vector_prev = compute_perpendicular_projection(previous_pos, desired_pos,
        #                                                              branch_vector + desired_pos)
        # # Get vector for current orientation of end effector
        # rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # rot_mat_prev = np.array(self.con.getMatrixFromQuaternion(previous_or)).reshape(3, 3)
        # # Initial vectors
        # init_vector = np.array([0, 0, 1])
        # camera_vector = rot_mat.dot(init_vector)
        # camera_vector_prev = rot_mat_prev.dot(init_vector)
        # OFFSET = np.array([0, 0, 0])
        # self.con.removeUserDebugItem(self.debug_cur_point)
        # self.con.removeUserDebugItem(self.debug_des_point)
        # self.debug_des_point = self.con.addUserDebugLine(achieved_pos + OFFSET,
        #                                                  achieved_pos + OFFSET + perpendicular_vector, [1, 0, 0], 2)
        # self.debug_cur_point = self.con.addUserDebugLine(achieved_pos, achieved_pos + 0.1 * camera_vector,
        #                                                  [0, 1, 0], 1)
        # orientation_reward_prev = np.dot(camera_vector_prev, perpendicular_vector_prev) / (
        #         np.linalg.norm(camera_vector_prev) * np.linalg.norm(perpendicular_vector_prev))
        # orientation_reward = np.dot(camera_vector, perpendicular_vector) / (
        #         np.linalg.norm(camera_vector) * np.linalg.norm(perpendicular_vector))
        # # print("Pointing Orientation reward: ", orientation_reward - orientation_reward_prev, orientation_reward)
        #
        orientation_reward_prev = self.compute_pointing_cos_sim(previous_pos, desired_pos, previous_or, branch_vector)
        orientation_reward = self.compute_pointing_cos_sim(achieved_pos, desired_pos, achieved_or, branch_vector)
        # print("Pointing Orientation reward: ", orientation_reward - orientation_reward_prev, orientation_reward)
        return (abs(orientation_reward) - abs(orientation_reward_prev)), abs(orientation_reward)

    def compute_pointing_cos_sim(self, achieved_pos: NDArray[Shape['3, 1'], Float],
                                            desired_pos: NDArray[Shape['3, 1'], Float],
                                            achieved_or: NDArray[Shape['4, 1'], Float],
                                            branch_vector: NDArray[Shape['3, 1'], Float]) -> Tuple[float, float]:
        # Orientation reward is computed as the dot product between the current orientation and the perpendicular vector to the end effector and goal pos vector
        # This is to encourage the end effector to be perpendicular to the branch

        # Perpendicular vector to branch vector
        perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector + desired_pos)
        rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # Initial vectors
        init_vector = np.array([0, 0, 1])
        camera_vector = rot_mat.dot(init_vector)
        pointing_cos_sim = np.dot(camera_vector, perpendicular_vector) / (
                np.linalg.norm(camera_vector) * np.linalg.norm(perpendicular_vector))
        # print("Pointing Orientation reward: ", orientation_reward - orientation_reward_prev, orientation_reward)
        return pointing_cos_sim

    def compute_perpendicular_orientation_reward(self, achieved_or: NDArray[Shape['4, 1'], Float],
                                                 previous_or: NDArray[Shape['4, 1'], Float],
                                                 branch_vector: NDArray[Shape['3, 1'], Float]):
        # Orientation reward is computed as the dot product between the current orientation and the perpendicular
        # vector to the end effector and goal pos vector This is to encourage the end effector to be perpendicular to
        # the branch

        # Perpendicular vector to branch vector
        # perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector + desired_pos)
        # perpendicular_vector_prev = compute_perpendicular_projection(previous_pos, desired_pos,
        #                                                              branch_vector + desired_pos)
        # Get vector for current orientation of end effector
        # rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # rot_mat_prev = np.array(self.con.getMatrixFromQuaternion(previous_or)).reshape(3, 3)
        # # Initial vectors
        # init_vector = np.array([1, 0, 0])
        # camera_vector = rot_mat.dot(init_vector)
        # camera_vector_prev = rot_mat_prev.dot(init_vector)
        # OFFSET = np.array([0, 0, 0])
        # # self.con.removeUserDebugItem(self.debug_cur_perp)
        # # self.con.removeUserDebugItem(self.debug_des_perp)
        # # self.debug_des_perp = self.con.addUserDebugLine(achieved_pos, achieved_pos + branch_vector, [1, 1, 0], 2)
        # # self.debug_cur_perp = self.con.addUserDebugLine(achieved_pos, achieved_pos + 0.1 * camera_vector,
        # #                                               [0, 1, 1], 1)
        # # Check antiparallel case as well
        # orientation_reward_prev = np.dot(camera_vector_prev, branch_vector) / (
        #         np.linalg.norm(camera_vector_prev) * np.linalg.norm(branch_vector))
        # orientation_reward = np.dot(camera_vector, branch_vector) / (
        #         np.linalg.norm(camera_vector) * np.linalg.norm(branch_vector))
        cosine_sim_perp_prev = self.compute_perpendicular_cos_sim(previous_or, branch_vector)
        cosine_sim_perp = self.compute_perpendicular_cos_sim(achieved_or, branch_vector)
        # print("Orientation reward: ", abs(orientation_reward) - abs(orientation_reward_prev), orientation_reward)
        # print("Perpendicular Orientation reward: ", abs(cosine_sim_perp) - abs(cosine_sim_perp_prev), abs(cosine_sim_perp))
        return (abs(cosine_sim_perp) - abs(cosine_sim_perp_prev)), abs(cosine_sim_perp)

    def compute_perpendicular_cos_sim(self, achieved_or: NDArray[Shape['4, 1'], Float],
                                                 branch_vector: NDArray[Shape['3, 1'], Float]):
        # Orientation reward is computed as the dot product between the current orientation and the perpendicular
        # vector to the end effector and goal pos vector This is to encourage the end effector to be perpendicular to
        # the branch

        # Perpendicular vector to branch vector
        # perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector + desired_pos)
        # perpendicular_vector_prev = compute_perpendicular_projection(previous_pos, desired_pos,
        #                                                              branch_vector + desired_pos)
        # Get vector for current orientation of end effector
        rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3, 3)
        # Initial vectors
        init_vector = np.array([1, 0, 0])
        camera_vector = rot_mat.dot(init_vector)
        # Check antiparallel case as well
        cosine_sim_perp = np.dot(camera_vector, branch_vector) / (
                np.linalg.norm(camera_vector) * np.linalg.norm(branch_vector))
        return cosine_sim_perp

    def compute_reward(self, desired_goal: NDArray[Shape['3, 1'], Float], achieved_pose: NDArray[Shape['6, 1'], Float],
                       previous_pose: NDArray[Shape['6, 1'], Float], singularity: bool, info: dict) -> Tuple[float, dict]:

        reward = float(0)
        reward_info = {}
        # Give rewards better names, and appropriate scales
        achieved_pos = achieved_pose[:3]
        achieved_or = achieved_pose[3:]
        desired_pos = desired_goal
        previous_pos = previous_pose[:3]
        previous_or = previous_pose[3:]
        # There will be two different types of achieved positions, one for the end effector and one for the camera

        self.collisions_acceptable = 0
        self.collisions_unacceptable = 0
        self.delta_movement = float(goal_reward(achieved_pos, previous_pos, desired_pos))
        self.debug_line = self.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0, 0, 1], 20)
        self.debug_line_1 = self.con.addUserDebugLine(previous_pos, self.desired_pos, [0, 0, 1], 20)

        self.target_dist = float(goal_distance(achieved_pos, desired_pos))

        movement_reward = self.delta_movement * self.movement_reward_scale
        # print("Movement reward: ", self.delta_movement)
        reward_info['movement_reward'] = movement_reward
        reward += movement_reward

        distance_reward = (np.exp(-self.target_dist * 5) * self.distance_reward_scale)
        reward_info['distance_reward'] = distance_reward
        reward += distance_reward
        # if self.target_dist<0.2:

        self.pointing_orientation_reward_unscaled, self.orientation_point_value = self.compute_pointing_orientation_reward(
            achieved_pos, desired_pos,
            achieved_or, previous_pos,
            previous_or,
            self.tree_goal_branch)
        # else:
        #     self.orientation_reward_unscaled = 0
        #     self.cosine_sim = 0
        pointing_orientation_reward = (self.pointing_orientation_reward_unscaled *
                                       self.pointing_orientation_reward_scale)

        reward_info['pointing_orientation_reward'] = pointing_orientation_reward
        reward += pointing_orientation_reward

        self.perpendicular_orientation_reward_unscaled, self.orientation_perp_value = (
            self.compute_perpendicular_orientation_reward(
                achieved_or, previous_or, self.tree_goal_branch))
        # else:
        #     self.orientation_reward_unscaled = 0
        #     self.cosine_sim = 0
        perpendicular_orientation_reward = (self.perpendicular_orientation_reward_unscaled *
                                            self.perpendicular_orientation_reward_scale)

        reward_info['perpendicular_orientation_reward'] = perpendicular_orientation_reward
        reward += perpendicular_orientation_reward
        # print('Orientation reward: ', orientation_reward)
        # camera_vector = camera_vector/np.linalg.norm(camera_vector)
        # perpendicular_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

        condition_number = self.get_condition_number()
        condition_number_reward = 0
        if singularity:
            print('Too high condition number!')
            self.singularity_terminated = False
            condition_number_reward = 0 # TODO: Replace with an input argument
        elif self.terminate_on_singularity:
            condition_number_reward = np.abs(1 / condition_number) * self.condition_reward_scale
        reward += condition_number_reward
        reward_info['condition_number_reward'] = condition_number_reward

        is_collision, collision_info = self.check_collisions()
        is_success_collision = self.check_success_collision()
        # print('Success collision: ', is_success_collision, is_collision)
        terminate_reward = 0
        # print(self.orientation_perp_value, self.orientation_point_value, self.target_dist)
        # TODO: Point of is collision within a target distance is substitute of touch the required branch. Can you make it better?
        if is_success_collision and self.target_dist < self.learning_param:
            print(collision_info)
            if (self.orientation_perp_value > 0.7) and (
                    self.orientation_point_value > 0.7):  # and approach_velocity < 0.05:
                self.terminated = True
                terminate_reward = 1 * self.terminate_reward_scale
                reward += terminate_reward
                print('Successful!')
            else:

                # self.wrong_success = True
                # terminate_reward = -1
                # reward += -0.3
                print('Unsuccessful!')

        reward_info['terminate_reward'] = terminate_reward

        # check collisions:
        collision_reward = 0
        if is_collision:
            if collision_info['collisions_acceptable']:
                collision_reward = 1 * self.collision_reward_scale
                self.collisions_acceptable += 1
                # print('Collision acceptable!')
            elif collision_info['collisions_unacceptable']:
                collision_reward = 1 * self.collision_reward_scale
                self.collisions_unacceptable += 1
                # print('Collision unacceptable!')

        reward += collision_reward
        reward_info['collision_reward'] = collision_reward

        slack_reward = 1 * self.slack_reward_scale
        reward_info['slack_reward'] = slack_reward
        reward += slack_reward

        # Minimize joint velocities
        velocity_mag = np.linalg.norm(self.joint_velocities)
        velocity_reward = velocity_mag  # -np.clip(velocity_mag, -0.1, 0.1)
        # reward += velocity_reward
        reward_info['velocity_reward'] = velocity_reward
        return reward, reward_info


def goal_distance(goal_a: NDArray[Shape['3, 1'], Float], goal_b: NDArray[Shape['3, 1'], Float]) -> float:
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def goal_reward(current: NDArray[Shape['3, 1'], Float], previous: NDArray[Shape['3, 1'], Float],
                target: NDArray[Shape['3, 1'], Float]):
    # Compute the reward between the previous and current goal.
    assert current.shape == previous.shape
    assert current.shape == target.shape
    diff_prev = goal_distance(previous, target)
    diff_curr = goal_distance(current, target)
    reward = diff_prev - diff_curr
    return reward
def goal_reward_projection(current: NDArray[Shape['3, 1'], Float], previous: NDArray[Shape['3, 1'], Float],
                target: NDArray[Shape['3, 1'], Float]):
    # Compute the reward between the previous and current goal.
    assert current.shape == previous.shape
    assert current.shape == target.shape
    # get parallel projection of current on prev
    projection = compute_parallel_projection_vector(current - target, previous - target)

    reward = np.linalg.norm(previous - target) - np.linalg.norm(projection)
    # print(np.linalg.norm(previous - target), np.linalg.norm(projection), np.linalg.norm(current - target))

    return reward

# x,y distance
def goal_distance2d(goal_a: NDArray[Shape['3, 1'], Float], goal_b: NDArray[Shape['3, 1'], Float]):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)


def compute_perpendicular_projection(a: NDArray[Shape['3, 1'], Float], b: NDArray[Shape['3, 1'], Float],
                                     c: NDArray[Shape['3, 1'], Float]):
    ab = b - a
    bc = c - b
    projection = ab - np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection

def compute_parallel_projection_vector(ab: NDArray[Shape['3, 1'], Float], bc: NDArray[Shape['3, 1'], Float]):
    projection = np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection

def compute_perpendicular_projection_vector(ab: NDArray[Shape['3, 1'], Float], bc: NDArray[Shape['3, 1'], Float]):
    projection = ab - np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection



#TODO: Make own file
class Tree:
    """ Class representing a tree mesh. It is used to sample points on the surface of the tree."""

    def __init__(self, env: PruningEnv, urdf_path: str, obj_path: str,
                 pos: NDArray[Shape['3,1'], Float] = np.array([0, 0, 0]),
                 orientation: NDArray[Shape['4,1'], Float] = np.array([0, 0, 0, 1]),
                 num_points: Optional[int] = None, scale: int = 1) -> None:
        self.urdf_path = urdf_path
        self.env = env
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.tree_obj = pywavefront.Wavefront(obj_path, create_materials=True, collect_faces=True)
        self.vertex_and_projection = []
        self.transformed_vertices = list(map(self.transform_obj_vertex, self.tree_obj.vertices))
        self.projection_mean = np.array(0.)
        self.projection_std = np.array(0.)
        self.projection_sum_x = np.array(0.)
        self.projection_sum_x2 = np.array(0.)
        self.init_xyz = self.env.get_current_pose(self.env.end_effector_index)[0]
        # if pickled file exists load and return
        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        #TODO: Add reset variable so that even if present it recomputes
        if not os.path.exists('./pkl2/' + str(path_component[3])):
            os.makedirs('./pkl2/' + str(path_component[3]))
        pkl_path = './pkl2/' + str(path_component[3]) + '/' + str(path_component[-1][:-5]) + '_reachable_points.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.reachable_points = pickle.load(f)
            print('Loaded reachable points from pickle file ', self.urdf_path[:-5] + '_reachable_points.pkl')
            print("Number of reachable points: ", len(self.reachable_points))
            # # Uncomment to visualize sphere at each reachable point
            # self.active()
            # for num, i in enumerate(self.reachable_points):
            #     # print(i)
            #     if num % 10 == 0:
            #         visualShapeId = self.env.con.createVisualShape(self.env.con.GEOM_SPHERE, radius=0.005,
            #                                                        rgbaColor=[1, 0, 0, 1])
            #         self.sphereUid = self.env.con.createMultiBody(0.0, -1, visualShapeId, [i[0][0], i[0][1], i[0][2]],
            #                                                       [0, 0, 0, 1])
            #         point = [i[0][0], i[0][1], i[0][2]]
            #         normal_vec = i[2]
            #         self.debug_branch = self.env.con.addUserDebugLine(point,
            #                                                       point + 50 * normal_vec / np.linalg.norm(normal_vec),
            #                                                       [1, 0, 0], 200)

            return
        # Find the two longest edges of the face
        # Add their mid-points and perpendicular projection to the smallest side as a point and branch

        for num, face in enumerate(self.tree_obj.mesh_list[0].faces):
            # Order the sides of the face by length
            ab = (
                face[0], face[1],
                np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[1]]))
            ac = (
                face[0], face[2],
                np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[2]]))
            bc = (
                face[1], face[2],
                np.linalg.norm(self.transformed_vertices[face[1]] - self.transformed_vertices[face[2]]))
            sides = [ab, ac, bc]
            normal_vec = np.cross(self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]],
                                  self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])
           #Only front facing faces
            if np.dot(normal_vec, [0,1,0]) < 0:
                continue
                print(" is not pointing upwards")
            #     continue
            # argsort sorts in ascending order
            sorted_sides = np.argsort([x[2] for x in sides])
            ac = sides[sorted_sides[2]]
            ab = sides[sorted_sides[1]]
            bc = sides[sorted_sides[0]]
            # |a
            # |\
            # | \
            # |  \
            # |   \
            # |    \
            # b______\c
            perpendicular_projection = compute_perpendicular_projection_vector(
                self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]],
                self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])

            # self.vertex_and_projection.append(
            #     ((self.transformed_vertices[ac[0]] + self.transformed_vertices[ac[1]]) / 2, perpendicular_projection))
            self.vertex_and_projection.append(
                ((self.transformed_vertices[ab[0]] + self.transformed_vertices[ab[1]]) / 2, perpendicular_projection, normal_vec))

            # This projection mean is used to filter corner/flushed faces which do not correspond to a branch
            # Twice as each face has two point aded
            self.projection_sum_x += np.linalg.norm(perpendicular_projection)
            self.projection_sum_x2 += np.linalg.norm(perpendicular_projection) ** 2
            # if num % 100 == 0:
            #     self.active()
            #     visualShapeId = self.env.con.createVisualShape(self.env.con.GEOM_SPHERE, radius=0.001,
            #                                                    rgbaColor=[1, 0, 0, 1])
            #     self.sphereUid = self.env.con.createMultiBody(0.0, -1, visualShapeId, [self.vertex_and_projection[-1][0][0],self.vertex_and_projection[-1][0][1],self.vertex_and_projection[-1][0][2]],
            #                                                   [0, 0, 0, 1])
            #     point = self.vertex_and_projection[-1][0]
            #     # point = (np.array(ac) + np.array(bc) + np.array(ab))/3  #(ac+bc+ab)/3
            #     self.debug_branch = self.env.con.addUserDebugLine(point,
            #                                                   point + 50 * normal_vec / np.linalg.norm(normal_vec),
            #                                                   [1, 0, 0], 200)

                # input("Press Enter to continue...")
        self.projection_mean = self.projection_sum_x / len(self.vertex_and_projection)
        self.projection_std = np.sqrt(
            self.projection_sum_x2 / len(self.vertex_and_projection) - self.projection_mean ** 2)
        self.num_points = num_points
        self.get_reachable_points()
        # dump reachable points to file using pickle

        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        # if pkl path exists else create
        # Uncomment to visualize sphere at each reachable point
        # self.active()
        # for num,i in enumerate(self.reachable_points):
        #     # print(i)
        #     if num%1 == 0:
        #         visualShapeId = self.env.con.createVisualShape(self.env.con.GEOM_SPHERE, radius=0.005,
        #                                                        rgbaColor=[1, 0, 0, 1])
        #         self.sphereUid = self.env.con.createMultiBody(0.0, -1, visualShapeId, [i[0][0], i[0][1], i[0][2]],
        #                                               [0, 0, 0, 1])
        #     input("Press Enter to continue...")
        print("Saving pickle, num points ", len(self.reachable_points))
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.reachable_points, f)

    def active(self):
        print('Loading tree from ', self.urdf_path)
        self.supports = self.env.con.loadURDF(SUPPORT_AND_POST_PATH, [0, -0.7, 0],
                                              list(self.env.con.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])),
                                              globalScaling=1)
        self.tree_urdf = self.env.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        self.env.con.removeBody(self.tree_urdf)
        self.env.con.removeBody(self.supports)

    def transform_obj_vertex(self, vertex: List) -> NDArray[Shape['3, 1'], Float]:
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation
        vertex_w_transform = self.env.con.multiplyTransforms(self.pos, self.orientation, vertex_pos, vertex_orientation)
        return np.array(vertex_w_transform[0])

    def is_reachable(self, vertice: Tuple[NDArray[Shape['3, 1'], Float], NDArray[Shape['3, 1'], Float]]) -> bool:
        ur5_base_pos = self.init_xyz  # np.array(self.env.get_current_pose(self.env.end_effector_index)[0])
        if "envy" in self.urdf_path:
            if abs(vertice[0][0]) < 0.05:
                return False
        # elif "ufo" in self.urdf_path:
        #     if vertice[0][2] < 0.1:
        #         return False
        dist = np.linalg.norm(ur5_base_pos - vertice[0], axis=-1)
        projection_length = np.linalg.norm(vertice[1])
        if dist >= 1 or not (projection_length > self.projection_mean + self.projection_std):
            return False
        # Check if start point reachable

        j_angles = self.env.calculate_ik((vertice[0][0], self.env.init_pos[0][1], self.env.init_pos[0][2]),
                                         self.env.init_pos[1])

        self.env.set_joint_angles(j_angles)
        for i in range(100):
            self.env.con.stepSimulation()
        # ee_pos, _ = self.env.get_current_pose(self.env.end_effector_index)
        start_condition_number = self.env.get_condition_number()
        # print(start_condition_number)
        # Check if goal reachable
        j_angles = self.env.calculate_ik(vertice[0], None)

        self.env.set_joint_angles(j_angles)
        for i in range(100):
            self.env.con.stepSimulation()
        ee_pos, _ = self.env.get_current_pose(self.env.end_effector_index)
        dist = np.linalg.norm(np.array(ee_pos) - vertice[0], axis=-1)
        condition_number = self.env.get_condition_number()
        if dist <= 0.05 and condition_number < 40 and start_condition_number < 20:
            return True
        return False

    def get_reachable_points(self) -> List[Tuple[NDArray[Shape['3, 1'], Float], NDArray[Shape['3, 1'], Float]]]:
        self.reachable_points = list(filter(lambda x: self.is_reachable(x), self.vertex_and_projection))
        # self.reachable_points = [np.array(i[0][0:3]) for i in self.reachable_points]
        np.random.shuffle(self.reachable_points)
        if self.num_points:
            self.reachable_points = self.reachable_points[0:self.num_points]
        print("Number of reachable points: ", len(self.reachable_points))

        return self.reachable_points

    @staticmethod
    def make_trees_from_folder(env: PruningEnv, trees_urdf_path: str, trees_obj_path: str, pos: NDArray,
                               orientation: NDArray, scale: int, num_points: int, num_trees: int):
        trees: List[Tree] = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path + '/*.urdf')),
                             sorted(glob.glob(trees_obj_path + '/*.obj'))):
            if len(trees) >= num_trees:
                break
            trees.append(Tree(env, urdf_path=urdf, obj_path=obj, pos=pos, orientation=orientation, scale=scale,
                              num_points=num_points))
        return trees
