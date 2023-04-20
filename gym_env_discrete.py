
import pywavefront
import numpy as np
import random
import time
import numpy as np
import sys
from gym import spaces
import gym
from pybullet_utils import bullet_client as bc
import os
import math
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple
from enum import Enum
import glob

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"


# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def goal_reward(current, previous, target):
    assert current.shape == previous.shape
    assert current.shape == target.shape
    diff_prev = goal_distance(previous, target)
    diff_curr = goal_distance(current, target)
    reward = diff_prev - diff_curr
    return reward

# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

class Tree():
    def __init__(self, env, urdf_path, obj_path, pos = np.array([0,0,0]), orientation = np.array([0,0,0,1]), num_points = None, scale = 1) -> None:
        self.urdf_path = urdf_path
        self.env = env
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.tree_obj = pywavefront.Wavefront(obj_path, collect_faces=True)
        self.transformed_vertices = list(map(self.transform_obj_vertex, self.tree_obj.vertices))
        ur5_base_pos, _ = self.env.con.getBasePositionAndOrientation(self.env.ur5)
        self.num_points = num_points
        self.get_reachable_points(ur5_base_pos)

    def active(self):
        self.tree_urdf = self.env.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        self.env.con.removeBody(self.tree_urdf)

    def transform_obj_vertex(self, vertex):
        vertex_pos = np.array(vertex[0:3])*self.scale
        vertex_orientation = [0,0,0,1] #Dont care about orientation
        vertex_w_transform = np.array(self.env.con.multiplyTransforms(self.pos, self.orientation, vertex_pos, vertex_orientation))
        return vertex_w_transform

    def is_reachable(self, vertice, ur5_base_pos):
        ur5_base_pos = np.array(ur5_base_pos)
        vertice = vertice[0]
        dist=np.linalg.norm(ur5_base_pos - vertice, axis=-1)
        if dist <= 1. and vertice[2]>0.2: #Make it hyperparameter
            return True
        return False

    def get_reachable_points(self, ur5_base_pos):
        self.reachable_points = list(filter(lambda x: self.is_reachable(x, ur5_base_pos), self.transformed_vertices))
        self.reachable_points = [np.array(i[0][0:3]) for i in self.reachable_points]
        np.random.shuffle(self.reachable_points)
        if self.num_points:
            self.reachable_points = self.reachable_points[0:self.num_points]
        return 

    @staticmethod
    def make_list_from_folder(env, trees_urdf_path, trees_obj_path, pos, orientation, scale, num_points):
        trees = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path+'/*.urdf')), sorted(glob.glob(trees_obj_path+'/*.obj'))):
            trees.append(Tree(env, urdf_path=urdf, obj_path=obj, pos=pos, orientation = orientation, scale=scale, num_points=num_points))

        return trees
   
        
# class action(Enum):
#     up = 1
#     down = 2
#     left = 3
#     right = 4
#     forward = 5
#     backward = 6
#     roll_up = 7
#     roll_down = 8
#     pitch_up = 9
#     pitch_down = 10
#     yaw_up = 11
#     yaw_down = 12

class ur5GymEnv(gym.Env):
    def __init__(self,
                 renders=False,
                 maxSteps=100,
                 learning_param=0.05,
                 tree_urdf_path = None,
                 tree_obj_path = None,
                 width = 224,
                 height = 224,
                 eval = False,
                 num_points = None, 
                 action_dim = 12,
                 name = "ur5GymEnv"):
        super(ur5GymEnv, self).__init__()
        self.renders = renders
        self.eval = eval
        assert tree_urdf_path != None
        assert tree_obj_path != None

        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        # setup pybullet sim:
        if self.renders:
            self.con = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.con = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.con.setTimeStep(5./240.)
        self.con.setGravity(0,0,-10)
        self.con.setRealTimeSimulation(False)
      
        self.con.resetDebugVisualizerCamera( cameraDistance=1.06, cameraYaw=-120.3, cameraPitch=-12.48, cameraTargetPosition=[-0.3,-0.06,0.4])
      
        
        self.observation_space = spaces.Dict({ 
            'depth': spaces.Box(low=-1.,
                     high=1.0,
                     shape=(1, 224, 224), dtype=np.float32),\
            'goal_pos': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (3,), dtype=np.float32),
            'cur_pos': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (3,), dtype=np.float32),
            'cur_or': spaces.Box(low = -2,
                        high = 2,
                        shape = (4,), dtype=np.float32)
                        })
        
        self.reset_counter = 0
        self.randomize_tree_count = 5

        # setup robot arm:
        self.end_effector_index = 7
        flags = self.con.URDF_USE_SELF_COLLISION
        self.ur5 = self.con.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = self.con.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        
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
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                self.con.setJointMotorControl2(self.ur5, info.id, self.con.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
        

        # collisionFilterGroup = 0
        # collisionFilterMask = 0
        # self.con.setCollisionFilterGroupMask(self.ur5, -1, collisionFilterGroup, collisionFilterMask)
        # object:
        self.init_joint_angles = (-1.57, -1.57,1.80,-3.14,-1.57, -1.57)
        self.set_joint_angles(self.init_joint_angles)
        self.collisions = 0
        
        self.near_val = 0.01
        self.far_val = 3
        self.height = height
        self.width = width
        self.proj_mat = self.con.computeProjectionMatrixFOV(
            fov=42, aspect = width / height, nearVal=self.near_val,
            farVal=self.far_val)
        
        # step simualator:
        for i in range(1000):
            self.con.stepSimulation()

        self.tree_point_pos = [1, 0, 0] # initial object pos
        self.sphereUid = -1

        self.name = name
       
        self.action_dim = action_dim
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.observation = {}
        self.previous_pose = (np.array(0), np.array(0))
        self.learning_param = learning_param
        self.step_size = 0.05
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(self.action_dim,), dtype=np.float32)
        self.joint_velocities = [0,0,0,0,0,0]
        # self.action_space = spaces.Discrete(self.action_dim)
        # self.actions = {'+x':0,
        #                 '-x':1,
        #                 '+y' : 2,
        #                 '-y' : 3,
        #                 '+z' : 4,
        #                 '-z' : 5,
        #                 'pitch_+y' : 6,
        #                 'pitch_-y' : 7,
        #                 'yaw_+z' : 8,
        #                 'yaw_-z' : 9
        #                 }
        #                 # 'roll_+x' : 6,
        #                 # 'roll_-x': 7,}

        # self.rev_actions = {v: k for k,v in self.actions.items()}
       
        #Init trees
        self.trees = Tree.make_list_from_folder(self, self.tree_urdf_path, self.tree_obj_path, pos = np.array([0, -0.8, 0]), orientation=np.array([0,0,0,1]), scale=0.1, num_points = num_points)
        self.tree = random.sample(self.trees, 1)[0]
        self.tree.active()
       


    def set_joint_angles(self, joint_angles):
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
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )
    def set_joint_velocities(self, joint_velocities):
       #Set joint velocities using pybullet motor control
        velocities = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        maxForce = 500
        self.con.setJointMotorControlArray(self.ur5, 
        indexes, 
        controlMode=self.con.VELOCITY_CONTROL,
        targetVelocities = joint_velocities,
       )

    def calculate_joint_velocities_from_end_effector_velocity(self, end_effector_velocity):
        #Calculate joint velocities from end effector velocity
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0], self.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
        jacobian = np.vstack(jacobian)
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity)
        return joint_velocities

    def get_joint_angles(self):
        j = self.con.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints


    def check_collisions(self):
        collisions = self.con.getContactPoints(bodyA = self.ur5)#, linkIndexA=self.end_effector_index)
        for i in range(len(collisions)):
            # print("collision")
            if collisions[i][-6] < 0 :
                # print("[Collision detected!] {}, {}".format(collisions[i][-6], collisions[i][3], collisions[i][4]))
                return True
        return False


    def calculate_ik(self, position, orientation):
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
       
        joint_angles = self.con.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, orientation,
            jointDamping=[0.01]*6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges
        )
        return joint_angles


    def get_current_pose(self):
        linkstate = self.con.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[4], linkstate[1] #Position wrt end effector, orientation wrt COM
        return (position, orientation)

    def set_camera(self, pose, orientation):
        pose, orientation = self.get_current_pose()
        rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3,3)
		#Initial vectors
        init_camera_vector = np.array([1, 0, 0])#
        init_up_vector = np.array([0, 0,1]) #
        #Rotated vectors
        camera_vector = rot_mat.dot(init_camera_vector)
        up_vector = rot_mat.dot(init_up_vector)
        view_matrix = self.con.computeViewMatrix(pose, pose + 0.1 * camera_vector, up_vector)
       
        return self.con.getCameraImage(self.width, self.height, viewMatrix = view_matrix, projectionMatrix = self.proj_mat, renderer = self.con.ER_BULLET_HARDWARE_OPENGL)
        
    @staticmethod
    def seperate_rgbd_rgb_d(rgbd, h = 224, w = 224):
        rgb = rgbd[2][:,:,0:3].reshape(3,h,w)/255
        depth = rgbd[3]
        return rgb, depth

    @staticmethod
    def linearize_depth(depth, far_val, near_val):
        depth_linearized = near_val / (far_val - (far_val -near_val) * depth + 0.00000001)
        return depth_linearized
   
    def get_rgbd_at_cur_pose(self):
        cur_p = self.get_current_pose()
        rgbd = self.set_camera(cur_p[0], cur_p[1])
        rgb,  depth = self.seperate_rgbd_rgb_d(rgbd)
        depth = depth.astype(np.float32)
        depth = self.linearize_depth(depth, self.far_val, self.near_val) - 0.5
        return rgb, depth

    def reset(self):
        # print("RESET")
        self.stepCounter = 0
        self.terminated = False
        self.ur5_or = [0.0, 1/2*math.pi, 0.0]
        self.reset_counter+=1
        self.collisions = 0
        if self.reset_counter%self.randomize_tree_count == 0:
            print("RANDOM TREE")
            print(self.tree.urdf_path)
            self.tree.inactive()
            self.tree = random.sample(self.trees, 1)[0]
            self.tree.active()

        # Sample new point
        self.tree_point_pos = random.sample(self.tree.reachable_points,1)[0]
        self.con.removeBody(self.sphereUid)
        colSphereId = -1   
        visualShapeId = self.con.createVisualShape(self.con.GEOM_SPHERE, radius=.02,rgbaColor =[1,0,0,1])
        self.sphereUid = self.con.createMultiBody(0.0, colSphereId, visualShapeId, [self.tree_point_pos[0],self.tree_point_pos[1],self.tree_point_pos[2]], [0,0,0,1])
        #Remove and add tree
        self.tree.inactive()
        #Remove ur5 arm body
        self.con.removeBody(self.ur5)
        #Create new ur5 arm body
        flags = self.con.URDF_USE_SELF_COLLISION
        self.ur5 = self.con.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.set_joint_velocities([0]*6)
        self.set_joint_angles(self.init_joint_angles)
        self.tree.active()
        # step simualator:
        for i in range(1000):
            self.con.stepSimulation()

        # get obs and return:
        self.getExtendedObservation()
        # print("resetting")
      
        return self.observation


    def step(self, action, debug = False):
       
        # get current position:
        curr_p = self.get_current_pose()
        self.previous_pose = curr_p

        # calculate joint velocities:
        self.prev_joint_velocities = self.joint_velocities
        self.joint_velocities = self.calculate_joint_velocities_from_end_effector_velocity(action)
        self.max_joint_velocities = np.array([6,6,6,6,6,6])
        if (self.joint_velocities > self.max_joint_velocities).any():
            print("Joint velocity too high ", self.joint_velocities)
            self.joint_velocities = np.array([0,0,0,0,0,0])

        # set joint velocities:
        self.set_joint_velocities(self.joint_velocities)
       
        # step simualator:
        for i in range(1):
            self.con.stepSimulation()
            # if self.renders: time.sleep(5./240.) 

        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.achieved_orient, self.desired_goal, self.previous_goal, None)
        done = self.my_task_done()
        
        info = {'is_success': False}
        if self.terminated == True:
            info['is_success'] = True
        self.stepCounter += 1
        info['episode'] = {"l": self.stepCounter,  "r": reward}

        return self.observation, reward, done, info

    def render(self, mode = "human"):
        
        cam_prop = (1024, 768)
        img_rgbd = self.con.getCameraImage(cam_prop[0], cam_prop[1])
        return img_rgbd[2]
     
    def close(self):
        self.con.disconnect()


    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):
        # sensor values:
        # js = self.get_joint_angles()

        tool_pos, tool_orient = self.get_current_pose()
        goal_pos = self.tree_point_pos

        self.achieved_goal = self.observation['cur_pos'] = np.array(tool_pos).astype(np.float32)
        self.desired_goal = self.observation['goal_pos'] = np.array(goal_pos).astype(np.float32)
        self.previous_goal = np.array(self.previous_pose[0])
        self.previous_orient = np.array(self.previous_pose[1])
        self.achieved_orient = self.observation['cur_or'] = np.array(tool_orient).astype(np.float32)
        self.rgb, self.depth = self.get_rgbd_at_cur_pose()
        self.observation['depth'] = np.expand_dims(self.depth.astype(np.float32), axis = 0)

    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c


    def compute_reward(self, achieved_goal, achieved_orient, desired_goal, achieved_previous_goal, info):
        reward = float(0)
        # Give rewards better names, and appropriate scales

        self.collisions = 0
        self.delta_movement = float(goal_reward(achieved_goal, achieved_previous_goal, desired_goal))
        self.target_dist = float(goal_distance(achieved_goal, desired_goal))

        scale = 10.
        movement_reward = self.delta_movement/(self.maxSteps*np.sqrt(3)*(5./240.))*scale #Mean around 0 -> Change in distance 0.036
        distance_reward = -self.target_dist/(self.maxSteps*np.sqrt(3)*(5./240.))*1/30
        reward += movement_reward
        reward += distance_reward

        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0], self.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        condition_number_reward = -condition_number/(self.maxSteps)
        reward += condition_number_reward
        
        terminate_reward = 0
        if self.target_dist < self.learning_param:  # and approach_velocity < 0.05:
            self.terminated = True
            terminate_reward = 1
            reward += terminate_reward
            print('Successful!')
        
        # check collisions:
        collision = False
        collision_reward = 0
        if self.check_collisions():
            collision_reward = -0.3/self.maxSteps*scale
            reward += collision_reward
            collision = True
            self.collisions+=1
            # print('Collision!')
        slack_reward = -0.1/self.maxSteps*scale
        reward+= slack_reward
        
        
        #if eval env use logger to plot all rewards seperately
        if self.name == "evalenv":
            self.logger.record("eval/movement_reward", movement_reward)
            self.logger.record("eval/distance_reward", distance_reward)
            self.logger.record("eval/terminate_reward", terminate_reward)
            self.logger.record("eval/collision_reward", collision_reward)
            self.logger.record("eval/slack_reward", slack_reward)
            self.logger.record("eval/total_reward", reward)
            self.logger.record("eval/condition_number_reward", condition_number_reward)

        return reward

