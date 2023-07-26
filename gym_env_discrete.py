
import pywavefront
import numpy as np
import random
import numpy as np
import gymnasium as gym

from gymnasium import spaces
# import gymnasium
from pybullet_utils import bullet_client as bc
import os
import math
import pybullet
import pybullet_data
from collections import namedtuple
import glob
import os

import pickle
        
ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"



class ur5GymEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,
                 renders=False,
                 maxSteps=500,
                 learning_param=0.05,
                 tree_urdf_path = None,
                 tree_obj_path = None,
                 tree_count = 9999,
                 width = 224,
                 height = 224,
                 eval = False,
                 num_points = None, 
                 action_dim = 12,
                 name = "ur5GymEnv",
                 terminate_on_singularity = True,
                 action_scale = 1,
                 movement_reward_scale = 1,
                 distance_reward_scale = 1,
                 condition_reward_scale = 1,
                 terminate_reward_scale = 1,
                 collision_reward_scale = 1,
                 slack_reward_scale = 1,
                 orientation_reward_scale = 1,
                 ):
        super(ur5GymEnv, self).__init__()
        assert tree_urdf_path != None
        assert tree_obj_path != None
        

        self.renders = renders
        self.render_mode = 'rgb_array'
        self.eval = eval
        self.terminate_on_singularity = terminate_on_singularity
        self.tree_urdf_path = tree_urdf_path
        self.tree_obj_path = tree_obj_path
        self.name = name
        self.action_dim = action_dim
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.tree_count = tree_count
        self.action_scale = action_scale
        self.movement_reward_scale = movement_reward_scale
        self.distance_reward_scale = distance_reward_scale
        self.condition_reward_scale = condition_reward_scale
        self.terminate_reward_scale = terminate_reward_scale
        self.collision_reward_scale = collision_reward_scale
        self.slack_reward_scale = slack_reward_scale
        self.orientation_reward_scale = orientation_reward_scale
        self.sum_reward = 0
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
            'desired_goal': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (3,), dtype=np.float32),
            'achieved_goal': spaces.Box(low = -5.,
                        high = 5.,
                        shape = (6,), dtype=np.float32),
            'joint_angles' : spaces.Box(low = -2*np.pi,
                        high = 2*np.pi,
                        shape = (6,), dtype=np.float32),
            'joint_velocities' : spaces.Box(low = -6,
                        high = 6,
                        shape = (6,), dtype=np.float32),
            'prev_action' : spaces.Box(low = -1., high = 1., 
                                       shape = (self.action_dim,), dtype=np.float32),
            'cosine_sim': spaces.Box(low = -1., high = 1.,shape = (1,), dtype=np.float32)
                        })
        self.prev_action = np.zeros(self.action_dim)
        self.grayscale = np.zeros((224,224))
        self.cosine_sim = 0.5
        self.reset_counter = 0
        self.randomize_tree_count = 1
        self.sphereUid = -1
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_dim,), dtype=np.float32)
        self.learning_param = learning_param
 
        # setup robot arm:
        self.setup_ur5_arm()
        self.reset_env_variables()
        #Camera parameters
        self.near_val = 0.01
        self.far_val = 3
        self.height = height
        self.width = width
        self.proj_mat = self.con.computeProjectionMatrixFOV(
            fov=42, aspect = width / height, nearVal=self.near_val,
            farVal=self.far_val)

        
        #Tree parameters
        self.tree_goal_pos = [1, 0, 0] # initial object pos
        self.tree_goal_branch = [0, 0, 0]
        pos = None
        scale = None
        if "envy" in self.tree_urdf_path:
            pos = np.array([0, -0.8, 0])
            scale = 0.1
        elif "ufo" in self.tree_urdf_path:
            pos = np.array([-0.5, -0.8, -0.3])
            scale = 0.05
        
        assert scale is not None
        assert pos is not None
        self.trees = Tree.make_list_from_folder(self, self.tree_urdf_path, self.tree_obj_path, pos = pos,\
                                                 orientation=np.array([0,0,0,1]), scale=scale, num_points = num_points, num_trees = self.tree_count)
        self.tree = random.sample(self.trees, 1)[0]
        self.tree.active()

        #Debug parameters
        self.debug_line = -1
        self.debug_cur_or = -1
        self.debug_des_or = -1
        self.debug_branch = -1
       
    def reset_env_variables(self):
        # Env variables that will change
        self.observation = {}
        self.stepCounter = 0
        self.sum_reward = 0
        self.terminated = False
        self.singularity_terminated = False
        self.collisions = 0
        
        


    def setup_ur5_arm(self):
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
    
        self.init_joint_angles = (-1.57, -1.57,1.80,-3.14,-1.57, -1.57)
        self.set_joint_angles(self.init_joint_angles)
        for i in range(1000):
            self.con.stepSimulation()
        self.init_pos = self.con.getLinkState(self.ur5, self.end_effector_index)
        self.joint_velocities = np.array([0,0,0,0,0,0]).astype(np.float32)
        self.joint_angles = self.init_joint_angles
        self.achieved_pos = self.get_current_pose()[0]
        self.previous_pose = np.array([0,0,0,0,0,0,0]).astype(np.float32)

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
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
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

    def set_camera(self):
        pose, orientation = self.get_current_pose()
        CAMERA_BASE_OFFSET = np.array([-0.0, 0., 0.])
        pose = pose + CAMERA_BASE_OFFSET
        rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3,3)
		#Initial vectors
        init_camera_vector = np.array([1, 0, 0])#
        init_up_vector = np.array([0, 0,1]) #
        #Rotated vectors
        camera_vector = rot_mat.dot(init_camera_vector)
        up_vector = rot_mat.dot(init_up_vector)
        view_matrix = self.con.computeViewMatrix(pose, pose + 0.1 * camera_vector, up_vector)
       
        return self.con.getCameraImage(self.width, self.height, viewMatrix = view_matrix, projectionMatrix = self.proj_mat, renderer = self.con.ER_BULLET_HARDWARE_OPENGL, flags = self.con.ER_NO_SEGMENTATION_MASK, lightDirection = [1,1,1])
        
    @staticmethod
    def seperate_rgbd_rgb_d(rgbd, h = 224, w = 224):
        rgb = np.array(rgbd[2]).reshape(h,w,4)/255
        rgb = rgb[:,:,:3]
        depth = np.array(rgbd[3]).reshape(h,w)
        return rgb, depth

    @staticmethod
    def linearize_depth(depth, far_val, near_val):
        depth_linearized = near_val / (far_val - (far_val -near_val) * depth + 0.00000001)
        return depth_linearized
   
    def get_rgbd_at_cur_pose(self):
        cur_p = self.get_current_pose()
        rgbd = self.set_camera()
        rgb,  depth = self.seperate_rgbd_rgb_d(rgbd)
        depth = depth.astype(np.float32)
        depth = self.linearize_depth(depth, self.far_val, self.near_val) - 0.5
        return rgb, depth

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        random.seed(seed)
        self.reset_env_variables()
        self.reset_counter+=1
        #Remove and add tree to avoid collisions with tree while resetting
        self.tree.inactive()
        self.con.removeBody(self.ur5)
        #Remove debug items
        self.con.removeUserDebugItem(self.debug_branch)
        self.con.removeUserDebugItem(self.debug_line)
        self.con.removeUserDebugItem(self.debug_cur_or)
        self.con.removeUserDebugItem(self.debug_des_or)
        #Sample new tree if reset_counter is a multiple of randomize_tree_count
        if self.reset_counter%self.randomize_tree_count == 0:
            self.tree = random.sample(self.trees, 1)[0]

        #Create new ur5 arm body
        self.setup_ur5_arm()
        
        # Sample new point
        random_point = random.sample(self.tree.reachable_points,1)[0]
        if "eval" in self.name:
            print('eval')
            self.con.removeBody(self.sphereUid)
            colSphereId = -1   
            visualShapeId = self.con.createVisualShape(self.con.GEOM_SPHERE, radius=0.02,rgbaColor =[1,0,0,1])
            self.sphereUid = self.con.createMultiBody(0.0, colSphereId, visualShapeId, [random_point[0][0],random_point[0][1],random_point[0][2]], [0,0,0,1])
        self.set_joint_angles(self.init_joint_angles)

        self.tree_goal_pos = random_point[0]
        self.tree_goal_branch = random_point[1]
        self.tree.active()

        #Add debug branch
        self.debug_branch = self.con.addUserDebugLine(self.tree_goal_pos - 50*self.tree_goal_branch,self.tree_goal_pos+ 50*self.tree_goal_branch, [1,0,0], 200)
      
        self.getExtendedObservation()
        info = {}
        #Make info analogous to one in step function
        return self.observation, info

    def step(self, action):
        # remove debug line
        self.con.removeUserDebugItem(self.debug_line)
        self.prev_action = action
        self.prev_grayscale = self.grayscale
        previous_pose = self.get_current_pose()
        #convert two tuples into one array

        self.previous_pose = np.hstack((previous_pose[0], previous_pose[1]))
        self.prev_joint_velocities = self.joint_velocities

        action = action*self.action_scale
        self.joint_velocities = self.calculate_joint_velocities_from_end_effector_velocity(action)
        self.set_joint_velocities(self.joint_velocities)
        
        for i in range(5):
            self.con.stepSimulation()
            # if self.renders: time.sleep(5./240.) 
        self.getExtendedObservation()
        
        reward, reward_infos = self.compute_reward(self.desired_pos, np.hstack((self.achieved_pos, self.achieved_or)),  None)
        self.sum_reward += reward
        self.debug_line = self.con.addUserDebugLine(self.achieved_pos, self.desired_pos, [0,0,1], 20)
        done, terminate_info = self.is_task_done()
        truncated = terminate_info['time_limit_exceeded']
        terminated = terminate_info['goal_achieved'] or terminate_info['singularity_achieved']
        
        infos = {'is_success': False, "TimeLimit.truncated": False}
        if terminate_info['time_limit_exceeded']:
            infos["TimeLimit.truncated"] = True
            infos["terminal_observation"] = self.observation
            
            infos['episode'] = {"l": self.stepCounter,  "r": self.sum_reward}
        
        if self.terminated == True:
            infos['is_success'] = True
            infos['episode'] = {"l": self.stepCounter,  "r": self.sum_reward}
        
        self.stepCounter += 1
        # infos['episode'] = {"l": self.stepCounter,  "r": reward}
        infos.update(reward_infos)
        # return self.observation, reward, done, infos
        #v26
        return self.observation, reward, terminated, truncated, infos

    def render(self, mode = "rgb_array"):
        size = [960, 720]
        view_matrix = self.con.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.3,-0.06,0.4], distance=1.06, yaw=-120.3, pitch=-12.48, roll=0, upAxisIndex=2)
        proj_matrix = self.con.computeProjectionMatrixFOV(fov=60, aspect=float(size[0]) /size[1], nearVal=0.1, farVal=100.0)
        img_rgbd = self.con.getCameraImage(size[0], size[1], view_matrix, proj_matrix, renderer = self.con.ER_BULLET_HARDWARE_OPENGL, flags = self.con.ER_NO_SEGMENTATION_MASK)
                                           #, renderer = self.con.ER_BULLET_HARDWARE_OPENGL)
        img_rgb,  _ = self.seperate_rgbd_rgb_d(img_rgbd, size[1], size[0])
        if mode == "human":
            import cv2
            cv2.imshow("img", (img_rgb*255).astype(np.uint8))
            cv2.waitKey(1)
      
        return img_rgb
     
    def close(self):
        self.con.disconnect()


    def getExtendedObservation(self):
        """
        The observations are the current position, the goal position, the current orientation, the current depth image, the current joint angles and the current joint velocities    
        """   
        tool_pos, tool_orient = self.get_current_pose()
        self.achieved_pos = np.array(tool_pos).astype(np.float32) 
        self.achieved_or = np.array(tool_orient).astype(np.float32) 
        

        self.desired_pos = np.array(self.tree_goal_pos).astype(np.float32)
        self.rgb, self.depth = self.get_rgbd_at_cur_pose()
        self.grayscale = self.rgb.mean(axis=2).astype(np.uint8)

        self.joint_angles = np.array(self.get_joint_angles()).astype(np.float32)

        init_pos = np.array(self.init_pos[0]).astype(np.float32)
        init_or = np.array(self.init_pos[1]).astype(np.float32)

        self.observation['achieved_goal'] = np.hstack((self.achieved_pos - init_pos, np.array(self.con.getEulerFromQuaternion(self.achieved_or)) - np.array(self.con.getEulerFromQuaternion(init_or))))
        self.observation['desired_goal'] = self.desired_pos - init_pos
        # print(np.array(self.con.getEulerFromQuaternion(self.achieved_or)) - np.array(self.con.getEulerFromQuaternion(init_or)))
        # if use_optical_flow:
        #     flow = cv2.calcOpticalFlowFarneback(prev=self.last_grayscale, next=grayscale, flow=None,
        #                                             pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        #                                             poly_n=5, poly_sigma=1.1, flags=0)
        #     flow_mag = np.linalg.norm(flow, axis=2)
        #     # flow_img = (255 * flow_mag / flow_mag.max())
        #     self.observation['depth'] = flow_mag
        # else:
        self.observation['depth'] = np.expand_dims(self.depth.astype(np.float32), axis = 0)

        self.observation['cosine_sim'] =  np.array(self.cosine_sim).astype(np.float32).reshape(1,-1)
        self.observation['joint_angles'] = self.joint_angles - self.init_joint_angles
        self.observation['joint_velocities'] = self.joint_velocities
        self.observation['prev_action'] = self.prev_action

    def is_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        time_limit_exceeded = self.stepCounter > self.maxSteps
        singularity_achieved = self.singularity_terminated
        goal_achieved = self.terminated
        c = (self.singularity_terminated == True or self.terminated == True or self.stepCounter > self.maxSteps)
        terminate_info = {"time_limit_exceeded": time_limit_exceeded, "singularity_achieved": singularity_achieved, "goal_achieved": goal_achieved}
        return c, terminate_info

    def get_condition_number(self):
        #get jacobian
        jacobian = self.con.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0], self.get_joint_angles(), [0,0,0,0,0,0], [0,0,0,0,0,0])
        jacobian = np.vstack(jacobian)
        condition_number = np.linalg.cond(jacobian)
        return condition_number
    
    def compute_orientation_reward(self, achieved_pos, desired_pos, achieved_or, previous_pos, previous_or, branch_vector):
        # Orientation reward is computed as the dot product between the current orientation and the perpedicular vector to the end effector and goal pos vector
        # This is to encourage the end effector to be perpendicular to the branch

        #Perpendicular vector to branch vector
        perpendicular_vector = compute_perpendicular_projection(achieved_pos, desired_pos, branch_vector+desired_pos)
        perpendicular_vector_prev = compute_perpendicular_projection(previous_pos, desired_pos, branch_vector+desired_pos)
        #Get vector for current orientation of end effector
        rot_mat = np.array(self.con.getMatrixFromQuaternion(achieved_or)).reshape(3,3)
        rot_mat_prev = np.array(self.con.getMatrixFromQuaternion(previous_or)).reshape(3,3)
		#Initial vectors
        init_vector = np.array([1, 0, 0])
        camera_vector = rot_mat.dot(init_vector)
        camera_vector_prev = rot_mat_prev.dot(init_vector)

        self.con.removeUserDebugItem(self.debug_cur_or)
        self.con.removeUserDebugItem(self.debug_des_or)
        self.debug_des_or = self.con.addUserDebugLine(achieved_pos, achieved_pos + perpendicular_vector, [1,0,0], 2)
        self.debug_cur_or = self.con.addUserDebugLine(self.achieved_pos, self.achieved_pos + 0.1 * camera_vector, [0, 1, 0], 1)
        orientation_reward_prev = np.dot(camera_vector_prev, perpendicular_vector_prev)/(np.linalg.norm(camera_vector_prev)*np.linalg.norm(perpendicular_vector_prev))
        orientation_reward = np.dot(camera_vector, perpendicular_vector)/(np.linalg.norm(camera_vector)*np.linalg.norm(perpendicular_vector))
        # print("Orientation reward: ", orientation_reward)
        return (orientation_reward - orientation_reward_prev), orientation_reward
       
    
    def compute_reward(self, desired_goal, achieved_goal, info):#achieved_pos, achieved_or, desired_pos, previous_pos, info):
        reward = float(0)
        reward_info = {}
        # Give rewards better names, and appropriate scales
        achieved_pos = achieved_goal[:3]
        achieved_or = achieved_goal[3:]
        desired_pos = desired_goal
        previous_pos = self.previous_pose[:3]
        previous_or = self.previous_pose[3:]
        self.collisions = 0
        self.delta_movement = float(goal_reward(achieved_pos, previous_pos, desired_pos))
        self.target_dist = float(goal_distance(achieved_pos, desired_pos))

        movement_reward = self.delta_movement*self.movement_reward_scale
        reward_info['movement_reward'] = movement_reward
        reward += movement_reward

        distance_reward = (np.exp(-self.target_dist*5)*self.distance_reward_scale)
        reward_info['distance_reward'] = distance_reward
        reward += distance_reward
        if self.target_dist<0.2:
            self.orientation_reward_unscaled, self.cosine_sim = self.compute_orientation_reward(achieved_pos, desired_pos, achieved_or, previous_pos, previous_or, self.tree_goal_branch)
        else:
            self.orientation_reward_unscaled = 0
            self.cosine_sim = 0
        orientation_reward = (self.orientation_reward_unscaled)*self.orientation_reward_scale

        reward_info['orientation_reward'] = orientation_reward
        reward += orientation_reward
        # print('Orientation reward: ', orientation_reward)
        # camera_vector = camera_vector/np.linalg.norm(camera_vector)
        # perpendicular_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)
       
        # print('Orientation reward: ', orientation_reward, np.arccos(camera_vector[0])*180/np.pi - np.arccos(perpendicular_vector[0])*180/np.pi, np.arccos(camera_vector[1])*180/np.pi - np.arccos(perpendicular_vector[1])*180/np.pi, np.arccos(camera_vector[2])*180/np.pi - np.arccos(perpendicular_vector[2])*180/np.pi)
      
      
        
        condition_number = self.get_condition_number()
        condition_number_reward = 0
        if condition_number > 50 or (self.joint_velocities > 5).any():
            print('Too high condition number!')
            self.singularity_terminated = True
            condition_number_reward = -3
        elif self.terminate_on_singularity:
            condition_number_reward = np.abs(1/condition_number)*self.condition_reward_scale
        reward += condition_number_reward
        reward_info['condition_number_reward'] = condition_number_reward
        terminate_reward = 0
        if self.target_dist < self.learning_param and (orientation_reward == 0 or self.cosine_sim>0.95):  # and approach_velocity < 0.05:
            self.terminated = True
            terminate_reward = 1*self.terminate_reward_scale
            reward += terminate_reward
            print('Successful!')
        reward_info['terminate_reward'] = terminate_reward
        
        # check collisions:
        collision_reward = 0
        if self.check_collisions():
            collision_reward = 1*self.collision_reward_scale
            self.collisions+=1

        reward += collision_reward
        reward_info['collision_reward'] = collision_reward
        
        slack_reward = 1*self.slack_reward_scale
        reward_info['slack_reward'] = slack_reward
        reward+= slack_reward

        #Minimize joint velocities
        velocity_mag = np.linalg.norm(self.joint_velocities)
        velocity_reward = velocity_mag#-np.clip(velocity_mag, -0.1, 0.1)
        #reward += velocity_rewarid
        reward_info['velocity_reward'] = velocity_reward
        return reward, reward_info



def goal_distance(goal_a, goal_b):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def goal_reward(current, previous, target):
    # Compute the reward between the previous and current goal.
    assert current.shape == previous.shape
    assert current.shape == target.shape
    diff_prev = goal_distance(previous, target)
    diff_curr = goal_distance(current, target)
    reward = diff_prev - diff_curr
    return reward

# x,y distance
def goal_distance2d(goal_a, goal_b):
    # Compute the distance between the goal and the achieved goal.
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

def compute_perpendicular_projection(a, b, c):
    ab = b - a
    bc = c - b
    projection = ab - np.dot(ab, bc)/np.dot(bc, bc) * bc
    return projection

def compute_perpendicular_projection_vector(ab, bc):
    projection = ab - np.dot(ab, bc)/np.dot(bc, bc) * bc
    return projection

class Tree():
    def __init__(self, env, urdf_path, obj_path, pos = np.array([0,0,0]), orientation = np.array([0,0,0,1]), num_points = None,  scale = 1) -> None:
        self.urdf_path = urdf_path
        self.env = env
        self.scale = scale
        self.pos = pos
        self.orientation = orientation
        self.tree_obj = pywavefront.Wavefront(obj_path, create_materials = True, collect_faces=True)
        self.vertex_and_projection = []
        self.transformed_vertices = list(map(self.transform_obj_vertex, self.tree_obj.vertices))
        self.projection_mean = 0
        #if pickled file exists load an return
        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        pkl_path = './pkl/' + str(path_component[3])+'/'+str(path_component[-1][:-5]) + '_reachable_points.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.reachable_points = pickle.load(f)
            print('Loaded reachable points from pickle file ', self.urdf_path[:-5] + '_reachable_points.pkl')
            print("Number of reachable points: ", len(self.reachable_points))
            return
        #Find the two longest edges of the face
        #Add their mid-points and perpendicular projection to the smallest side as a point and branch
        
        
        for face in self.tree_obj.mesh_list[0].faces:
            #Order the sides of the face by length
            ab = (face[0], face[1], np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[1]]))
            ac = (face[0], face[2], np.linalg.norm(self.transformed_vertices[face[0]] - self.transformed_vertices[face[2]]))
            bc = (face[1], face[2], np.linalg.norm(self.transformed_vertices[face[1]] - self.transformed_vertices[face[2]]))
            sides = [ab, ac, bc]
            #argsort sorts in ascending order
            sorted_sides = np.argsort([x[2] for x in sides])
            ac = sides[sorted_sides[2]]
            ab = sides[sorted_sides[1]]
            bc = sides[sorted_sides[0]]
            #|a
            #|\
            #| \
            #|  \
            #|   \
            #|    \
            #b______\c
            perpendicular_projection = compute_perpendicular_projection_vector(self.transformed_vertices[ac[0]] - self.transformed_vertices[ac[1]], self.transformed_vertices[bc[0]] - self.transformed_vertices[bc[1]])
                                                    
            self.vertex_and_projection.append(((self.transformed_vertices[ac[0]]+self.transformed_vertices[ac[1]])/2, perpendicular_projection))
            self.vertex_and_projection.append(((self.transformed_vertices[ab[0]]+self.transformed_vertices[ab[1]])/2, perpendicular_projection))
            self.projection_mean += np.linalg.norm(perpendicular_projection)
            self.projection_mean += np.linalg.norm(perpendicular_projection)
          
        self.projection_mean = self.projection_mean/len(self.vertex_and_projection)
        self.num_points = num_points
        self.get_reachable_points(self.env.ur5)
        # dump reachable points to file using pickle
        # self.active()
        # for i in self.reachable_points:
        #     print(i)
        #     visualShapeId = self.env.con.createVisualShape(self.env.con.GEOM_SPHERE, radius=0.02,rgbaColor =[1,0,0,1])
        #     self.sphereUid = self.env.con.createMultiBody(0.0, -1, visualShapeId, [i[0][0],i[0][1],i[0][2]], [0,0,0,1])

        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.reachable_points, f)

        
    def active(self):

        print('Loading tree from ', self.urdf_path)
        self.tree_urdf = self.env.con.loadURDF(self.urdf_path, self.pos, self.orientation, globalScaling=self.scale)

    def inactive(self):
        self.env.con.removeBody(self.tree_urdf)

    def transform_obj_vertex(self, vertex):
        vertex_pos = np.array(vertex[0:3])*self.scale
        vertex_orientation = [0,0,0,1] #Dont care about orientation
        vertex_w_transform = self.env.con.multiplyTransforms(self.pos, self.orientation, vertex_pos, vertex_orientation)
        return np.array(vertex_w_transform[0])

    def is_reachable(self, vertice, ur5):
        ur5_base_pos = np.array(self.env.get_current_pose()[0])
        if "envy" in self.urdf_path:
            if abs(vertice[0][0]) < 0.05:
                return False
        elif "ufo" in self.urdf_path:
            if vertice[0][2] < 0.1:
                return False
        dist=np.linalg.norm(ur5_base_pos - vertice[0], axis=-1)
        projection_length = np.linalg.norm(vertice[1])
        if dist >= 1 or projection_length < self.projection_mean* 0.7:
            return False
        j_angles = self.env.calculate_ik(vertice[0], None)
        self.env.set_joint_angles(j_angles)
        self.env.con.stepSimulation()
        ee_pos, _ = self.env.get_current_pose()
        dist=np.linalg.norm(np.array(ee_pos) - vertice[0], axis=-1)
        condition_number = self.env.get_condition_number()
        if dist <= 0.05 and condition_number < 20: 
            return True
        return False

    def get_reachable_points(self, ur5):
        self.reachable_points = list(filter(lambda x: self.is_reachable(x, ur5), self.vertex_and_projection))
        # self.reachable_points = [np.array(i[0][0:3]) for i in self.reachable_points]
        np.random.shuffle(self.reachable_points)
        if self.num_points:
            self.reachable_points = self.reachable_points[0:self.num_points]
        print("Number of reachable points: ", len(self.reachable_points))
       
        return self.reachable_points

    @staticmethod
    def make_list_from_folder(env, trees_urdf_path, trees_obj_path, pos, orientation, scale, num_points, num_trees ):
        trees = []
        for urdf, obj in zip(sorted(glob.glob(trees_urdf_path+'/*.urdf')), sorted(glob.glob(trees_obj_path+'/*.obj'))):
            if len(trees) >= num_trees:
                break
            trees.append(Tree(env, urdf_path=urdf, obj_path=obj, pos=pos, orientation = orientation, scale=scale, num_points=num_points))
        return trees
 
