"""
train_reach.py

Trains an AI (using the PPO algorithm) to control the Panda arm's joints
so its gripper reaches a randomly placed target point.

"""

from reach_env import PandaReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    """Creates one copy of the environment. We run several of these at
    once (in parallel) to use more of your CPU cores and train faster."""
    def _init():
        env = PandaReachEnv(render=False)  # no window = much faster
        env = Monitor(env)  # tracks episode rewards for us to see progress
        return env
    return _init


if __name__ == "__main__":
    # Run 4 simulations in parallel (matches a typical 4-core CPU).
    # If your laptop has fewer cores, or training feels too slow/heavy,
    # try lowering this to 2.
    NUM_PARALLEL_ENVS = 4
    env = SubprocVecEnv([make_env() for _ in range(NUM_PARALLEL_ENVS)])

    # "MlpPolicy" means the AI's brain is a small neural network (a
    # Multi-Layer Perceptron) that takes in the 13 observation numbers
    # and outputs the 7 action numbers.
    model = PPO("MlpPolicy", env, verbose=1)

    # Total timesteps = how many individual simulation steps the AI
    # practices across all parallel environments combined. 
    TOTAL_TIMESTEPS = 100_000
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps across {NUM_PARALLEL_ENVS} parallel environments...")
    print("Watch the 'ep_rew_mean' number in the printed table -- it should")
    print("trend upward (less negative) over time as the AI gets better.\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save("ppo_panda_reach")
    print("\nTraining complete. Model saved as ppo_panda_reach.zip")
    print("Run evaluate_reach.py next to watch it in action.")