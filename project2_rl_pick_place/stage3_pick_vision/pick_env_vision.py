import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2


class PandaPickVisionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    FOV = 60
    NEAR_PLANE = 0.01
    FAR_PLANE = 2.0
    CUBE_HALF_HEIGHT = 0.02

    def __init__(self, render=False):
        super().__init__()

        connection_mode = p.GUI if render else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.num_arm_joints = 7
        self.finger_joint_1 = 9
        self.finger_joint_2 = 10
        self.end_effector_index = 11
        self.max_steps = 300
        self.step_count = 0

        self.cube_half_extent = 0.02
        self.lift_success_height = 0.10

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_arm_joints + 1,), dtype=np.float32)
        self.max_joint_delta = 0.05

        obs_dim = self.num_arm_joints + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.robot_id = None
        self.cube_id = None
        self.home_angles = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.78]
        self.cube_start_height = 0.0
        self.cube_pos_estimate = np.array([0.5, 0.0, 0.05])

        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        for i, angle in enumerate(self.home_angles):
            p.resetJointState(self.robot_id, i, angle)
        p.resetJointState(self.robot_id, self.finger_joint_1, 0.04)
        p.resetJointState(self.robot_id, self.finger_joint_2, 0.04)
        p.changeDynamics(self.robot_id, self.finger_joint_1, lateralFriction=1.0)
        p.changeDynamics(self.robot_id, self.finger_joint_2, lateralFriction=1.0)

    def _spawn_cube(self):
        x = self.np_random.uniform(0.40, 0.55)
        y = self.np_random.uniform(-0.15, 0.15)
        z = self.cube_half_extent + 0.001
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.cube_half_extent] * 3, rgbaColor=[1, 0, 0, 1])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.cube_half_extent] * 3)
        cube_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
        )
        p.changeDynamics(cube_id, -1, lateralFriction=1.0, spinningFriction=0.1)
        return cube_id

    def _move_arm_directly(self, target_pos, target_orn, settle_steps=150):
       
     
        joint_angles = p.calculateInverseKinematics(
            self.robot_id, self.end_effector_index, target_pos, targetOrientation=target_orn,
            maxNumIterations=200,
        )
        for i in range(self.num_arm_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i])
        for _ in range(settle_steps):
            # Actively command the arm to STAY at this pose every step,
            # not just once -- this holds it steady against gravity.
            for i in range(self.num_arm_joints):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_angles[i], force=200)
            p.stepSimulation()

    def _get_wrist_camera_image(self):
        link_state = p.getLinkState(self.robot_id, self.end_effector_index, computeForwardKinematics=True)
        gripper_pos, gripper_orn = link_state[0], link_state[1]
        rot_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3, 3)
        forward_vec = rot_matrix.dot(np.array([0, 0, 1]))
        up_vec = rot_matrix.dot(np.array([1, 0, 0]))
        camera_offset = -forward_vec * 0.05
        camera_pos = np.array(gripper_pos) + camera_offset

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_pos + forward_vec * 0.5,
            cameraUpVector=up_vec,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            self.FOV, self.CAMERA_WIDTH / self.CAMERA_HEIGHT, self.NEAR_PLANE, self.FAR_PLANE
        )
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            self.CAMERA_WIDTH, self.CAMERA_HEIGHT, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_array = np.reshape(rgb_img, (self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 4))[:, :, :3].astype(np.uint8)
        depth_array = np.reshape(depth_img, (self.CAMERA_HEIGHT, self.CAMERA_WIDTH))
        return rgb_array, depth_array, view_matrix, proj_matrix

    def _detect_red_cube_pixel_and_mask(self, rgb_image):
        """Returns (centroid_pixel, mask) or (None, None).

        The mask is EVERY pixel identified as part of the cube -- not just
        the centroid. We use it to get a robust depth reading instead of
        trusting a single pixel.
        """
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 50])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 2]
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, None
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # Build a mask containing ONLY the largest contour's pixels (in
        # case there was other red noise elsewhere in the image)
        single_object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(single_object_mask, [largest], -1, 255, thickness=cv2.FILLED)
        return (u, v), single_object_mask

    def _detect_red_cube_pixel(self, rgb_image):
        """Kept for compatibility -- just the centroid, no mask."""
        pixel, _ = self._detect_red_cube_pixel_and_mask(rgb_image)
        return pixel

    def _get_robust_depth(self, depth_array, mask):
       
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=2)

        # If eroding shrank the object down to nothing (very small in
        # frame), fall back to the original mask rather than failing.
        if np.sum(eroded_mask) == 0:
            eroded_mask = mask

        masked_depths = depth_array[eroded_mask == 255]
        if len(masked_depths) == 0:
            return None
        return float(np.median(masked_depths))

    def _pixel_to_world(self, u, v, depth_value, view_matrix, proj_matrix):
        """depth_value is now passed in directly (a robust median reading),
        instead of being read from a single pixel inside this function."""
        far, near = self.FAR_PLANE, self.NEAR_PLANE
        real_depth = far * near / (far - (far - near) * depth_value)

        view = np.array(view_matrix).reshape(4, 4, order="F")
        proj = np.array(proj_matrix).reshape(4, 4, order="F")
        x_ndc = (2.0 * u / self.CAMERA_WIDTH) - 1.0
        y_ndc = 1.0 - (2.0 * v / self.CAMERA_HEIGHT)
        clip = np.array([x_ndc, y_ndc, 1.0, 1.0])
        inv_vp = np.linalg.inv(proj.dot(view))
        world_h = inv_vp.dot(clip)
        world = world_h[:3] / world_h[3]

        cam_pos = np.linalg.inv(view)[:3, 3]
        direction = (world - cam_pos)
        direction = direction / np.linalg.norm(direction)
        corrected_world = cam_pos + direction * real_depth
        return corrected_world

    def _estimate_cube_position_from_vision(self):
       
        down_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

        self._move_arm_directly([0.5, 0.0, 0.35], down_orientation)
        rgb, depth, view_matrix, proj_matrix = self._get_wrist_camera_image()
        pixel, mask = self._detect_red_cube_pixel_and_mask(rgb)

        if pixel is None:
            return np.array([0.475, 0.0, 0.02])

        robust_depth = self._get_robust_depth(depth, mask)
        surface_pos = self._pixel_to_world(pixel[0], pixel[1], robust_depth, view_matrix, proj_matrix)
        estimate_1 = surface_pos.copy()
        estimate_1[2] -= self.CUBE_HALF_HEIGHT

        closer_pos = [estimate_1[0], estimate_1[1], estimate_1[2] + 0.15]
        self._move_arm_directly(closer_pos, down_orientation)
        rgb2, depth2, view_matrix2, proj_matrix2 = self._get_wrist_camera_image()
        pixel2, mask2 = self._detect_red_cube_pixel_and_mask(rgb2)

        if pixel2 is not None:
            robust_depth_2 = self._get_robust_depth(depth2, mask2)
            surface_pos_2 = self._pixel_to_world(pixel2[0], pixel2[1], robust_depth_2, view_matrix2, proj_matrix2)
            estimate_2 = surface_pos_2.copy()
            estimate_2[2] -= self.CUBE_HALF_HEIGHT
            return estimate_2

        return estimate_1

    def _get_obs(self):
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_arm_joints)]
        gripper_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        finger_width = p.getJointState(self.robot_id, self.finger_joint_1)[0]

        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)
        is_touching = 1.0 if len(contacts) > 0 else 0.0

        # KEY DIFFERENCE from pick_env.py: use the one-time VISION ESTIMATE
        # instead of ground-truth cube position.
        obs = np.array(
            list(joint_angles) + list(gripper_pos) + list(self.cube_pos_estimate) + [finger_width, is_touching],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_scene()
        self.cube_id = self._spawn_cube()

        for _ in range(50):
            p.stepSimulation()

        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        self.cube_start_height = cube_pos[2]

        # --- The one-time vision look, replacing ground-truth lookup ---
        self.cube_pos_estimate = self._estimate_cube_position_from_vision()

        # Reset the arm back to its starting pose after the "look" -- the
        # RL policy should always start each episode from the same home
        # position, using the vision estimate as one of its inputs.
        for i, angle in enumerate(self.home_angles):
            p.resetJointState(self.robot_id, i, angle)
        p.resetJointState(self.robot_id, self.finger_joint_1, 0.04)
        p.resetJointState(self.robot_id, self.finger_joint_2, 0.04)

        self.step_count = 0
        self.has_touched_this_episode = False
        self.prev_lift_amount = 0.0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        current_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_arm_joints)]
        new_angles = [
            current_angles[i] + float(action[i]) * self.max_joint_delta
            for i in range(self.num_arm_joints)
        ]
        for i in range(self.num_arm_joints):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, new_angles[i], force=200)

        gripper_command = (float(action[self.num_arm_joints]) + 1.0) / 2.0 * 0.04
        p.setJointMotorControl2(self.robot_id, self.finger_joint_1, p.POSITION_CONTROL, gripper_command, force=60)
        p.setJointMotorControl2(self.robot_id, self.finger_joint_2, p.POSITION_CONTROL, gripper_command, force=60)

        p.stepSimulation()

        gripper_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_index)[0])

       
        true_cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        true_cube_pos = np.array(true_cube_pos)

        distance_to_cube = float(np.linalg.norm(gripper_pos - true_cube_pos))
        lift_amount = float(true_cube_pos[2] - self.cube_start_height)

        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)
        is_touching = len(contacts) > 0

        lift_progress = lift_amount - self.prev_lift_amount
        self.prev_lift_amount = lift_amount

        reward = 0.0
        reward += -distance_to_cube * 1.0
        if is_touching and not self.has_touched_this_episode:
            reward += 2.0
            self.has_touched_this_episode = True
        reward += lift_progress * 100.0
        reward += -0.005

        terminated = False
        if lift_amount > self.lift_success_height:
            reward += 50.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        observation = self._get_obs()
        info = {
            "distance_to_cube": distance_to_cube,
            "lift_amount": lift_amount,
            "is_touching": is_touching,
            "vision_estimate_error_mm": float(np.linalg.norm(self.cube_pos_estimate - true_cube_pos)) * 1000,
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)