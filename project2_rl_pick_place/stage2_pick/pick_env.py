import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class PandaPickEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

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

        # OBSERVATION: 7 joint angles + gripper xyz(3) + cube xyz(3) +
        # current gripper opening width(1) + is-touching-cube flag(1) = 15
        obs_dim = self.num_arm_joints + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.robot_id = None
        self.cube_id = None
        self.home_angles = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.78]
        self.cube_start_height = 0.0

        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        for i, angle in enumerate(self.home_angles):
            p.resetJointState(self.robot_id, i, angle)
        # Start with the gripper open
        p.resetJointState(self.robot_id, self.finger_joint_1, 0.04)
        p.resetJointState(self.robot_id, self.finger_joint_2, 0.04)

        # Grippy friction so the fingers can actually hold the cube once closed
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

    def _get_obs(self):
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_arm_joints)]
        gripper_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        finger_width = p.getJointState(self.robot_id, self.finger_joint_1)[0]

        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)
        is_touching = 1.0 if len(contacts) > 0 else 0.0

        obs = np.array(
            list(joint_angles) + list(gripper_pos) + list(cube_pos) + [finger_width, is_touching],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_scene()

        if self.cube_id is not None:
            pass  
        self.cube_id = self._spawn_cube()

       
        for _ in range(50):
            p.stepSimulation()

        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        self.cube_start_height = cube_pos[2]

        self.step_count = 0
        self.has_touched_this_episode = False
        self.prev_lift_amount = 0.0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        # Apply the 7 arm-joint actions, same as Stage 1
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
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)

        distance_to_cube = float(np.linalg.norm(gripper_pos - cube_pos))
        lift_amount = float(cube_pos[2] - self.cube_start_height)

        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)
        is_touching = len(contacts) > 0

        lift_progress = lift_amount - self.prev_lift_amount
        self.prev_lift_amount = lift_amount

        reward = 0.0
        reward += -distance_to_cube * 1.0       # closer to the cube = better
        if is_touching and not self.has_touched_this_episode:
            reward += 2.0                        # one-time bonus for first contact only
            self.has_touched_this_episode = True
        reward += lift_progress * 100.0           # reward NEW height gained, not height held
        reward += -0.005                          # tiny penalty per step, encourages efficiency

        terminated = False
        if lift_amount > self.lift_success_height:
            reward += 50.0                        #
            terminated = True

        truncated = self.step_count >= self.max_steps

        observation = self._get_obs()
        info = {"distance_to_cube": distance_to_cube, "lift_amount": lift_amount, "is_touching": is_touching}
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)