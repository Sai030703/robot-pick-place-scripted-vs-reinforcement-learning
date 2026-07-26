import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class PandaReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render=False):
        super().__init__()

       
        connection_mode = p.GUI if render else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.num_joints = 7
        self.max_steps = 200          # give up after this many steps per attempt
        self.step_count = 0
        self.success_threshold = 0.05  # within 5cm counts as "reached the target"
        self.end_effector_index = 11

        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.max_joint_delta = 0.05  # radians per step

        # OBSERVATION SPACE: 7 joint angles + 3 numbers for gripper position
        # (x,y,z) + 3 numbers for target position (x,y,z) = 13 numbers total.
        obs_dim = self.num_joints + 3 + 3
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.robot_id = None
        self.target_position = np.array([0.5, 0.0, 0.3])
        self.home_angles = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.78]

        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        for i, angle in enumerate(self.home_angles):
            p.resetJointState(self.robot_id, i, angle)

    def _get_obs(self):
        """Package up everything the AI is allowed to see this step."""
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        gripper_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        obs = np.array(
            list(joint_angles) + list(gripper_pos) + list(self.target_position),
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        """Called at the start of every new attempt ('episode')."""
        super().reset(seed=seed)
        self._setup_scene()

       
        x = self.np_random.uniform(0.35, 0.55)
        y = self.np_random.uniform(-0.2, 0.2)
        z = self.np_random.uniform(0.15, 0.4)
        self.target_position = np.array([x, y, z])

        self.step_count = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Called every single timestep. This is where the reward is decided."""
        self.step_count += 1

       
        current_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        new_angles = [
            current_angles[i] + float(action[i]) * self.max_joint_delta
            for i in range(self.num_joints)
        ]
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, new_angles[i], force=200)
        p.stepSimulation()

        gripper_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_index)[0])
        distance = float(np.linalg.norm(gripper_pos - self.target_position))

        
        reward = -distance

        terminated = False
        if distance < self.success_threshold:
            reward += 10.0  
            terminated = True

        truncated = self.step_count >= self.max_steps  # ran out of time this attempt

        observation = self._get_obs()
        info = {"distance": distance}
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)