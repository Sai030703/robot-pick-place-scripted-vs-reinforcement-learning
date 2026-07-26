import time
from pick_env import PandaPickEnv
from stable_baselines3 import PPO

env = PandaPickEnv(render=True)
model = PPO.load("ppo_panda_pick")

NUM_EPISODES = 5 

for episode in range(1, NUM_EPISODES + 1):
    observation, info = env.reset()
    done = False
    total_reward = 0.0
    max_lift_seen = 0.0

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        max_lift_seen = max(max_lift_seen, info["lift_amount"])
        time.sleep(1.0 / 60.0)

    success = max_lift_seen > env.lift_success_height
    print(f"Episode {episode}: total reward {total_reward:.2f}, "
          f"max lift height {max_lift_seen*1000:.1f} mm, "
          f"success: {success}")

env.close()