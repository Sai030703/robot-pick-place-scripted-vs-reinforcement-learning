from pick_env_vision import PandaPickVisionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        env = PandaPickVisionEnv(render=False)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    NUM_PARALLEL_ENVS = 4
    env = SubprocVecEnv([make_env() for _ in range(NUM_PARALLEL_ENVS)])

    model = PPO("MlpPolicy", env, verbose=1)

    TOTAL_TIMESTEPS = 1_000_000
    print(f"Starting vision-based pick training for {TOTAL_TIMESTEPS} timesteps...")
    print("Each episode reset now includes a short camera 'look' step, so")
    print("this may run a bit slower than the ground-truth version did.\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save("ppo_panda_pick_vision")
    print("\nTraining complete. Model saved as ppo_panda_pick_vision.zip")
    print("Run evaluate_pick_vision.py next to check results.")