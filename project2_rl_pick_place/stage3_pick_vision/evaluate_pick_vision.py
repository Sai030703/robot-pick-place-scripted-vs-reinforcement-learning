import time
from pick_env_vision import PandaPickVisionEnv
from stable_baselines3 import PPO

env = PandaPickVisionEnv(render=True)
model = PPO.load("ppo_panda_pick_vision")

NUM_EPISODES = 5
results = []

for episode in range(1, NUM_EPISODES + 1):
    observation, info = env.reset()
    done = False
    total_reward = 0.0
    max_lift_seen = 0.0
    vision_error = None

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        max_lift_seen = max(max_lift_seen, info["lift_amount"])
        vision_error = info["vision_estimate_error_mm"]  # same each step within an episode
        time.sleep(1.0 / 60.0)

    success = max_lift_seen > env.lift_success_height
    results.append({
        "episode": episode, "reward": total_reward, "lift_mm": max_lift_seen * 1000,
        "success": success, "vision_error_mm": vision_error,
    })
    print(f"Episode {episode}: reward {total_reward:.2f}, lift {max_lift_seen*1000:.1f} mm, "
          f"vision error {vision_error:.1f} mm, success: {success}")

env.close()

num_successes = sum(1 for r in results if r["success"])
success_rate = num_successes / NUM_EPISODES * 100
avg_vision_error = sum(r["vision_error_mm"] for r in results) / NUM_EPISODES

print("\n" + "=" * 55)
print(f"VISION-BASED MODEL SUMMARY over {NUM_EPISODES} episodes:")
print(f"  Success rate: {num_successes}/{NUM_EPISODES} ({success_rate:.1f}%)")
print(f"  Average vision position error: {avg_vision_error:.1f} mm")
print("  (Compare this success rate to your ground-truth pick_env.py result")
print("   to see the real cost of using camera perception instead of")
print("   privileged simulator state.)")
print("=" * 55)