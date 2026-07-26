"""
evaluate_reach.py

Loads the trained AI model and watches it attempt the reach task with a
visible PyBullet window.

"""

import time
from reach_env import PandaReachEnv
from stable_baselines3 import PPO

env = PandaReachEnv(render=True)  # render=True this time -- we want to WATCH it
model = PPO.load("ppo_panda_reach")

NUM_EPISODES = 5

for episode in range(1, NUM_EPISODES + 1):
    observation, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(1.0 / 60.0)  # slow down so it's watchable, not instant

    print(f"Episode {episode}: total reward {total_reward:.2f}, "
          f"final distance to target: {info['distance']*1000:.1f} mm")

env.close()