from pick_env import PandaPickEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        env = PandaPickEnv(render=False)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    NUM_PARALLEL_ENVS = 4
    env = SubprocVecEnv([make_env() for _ in range(NUM_PARALLEL_ENVS)])

    model = PPO("MlpPolicy", env, verbose=1)

  
    TOTAL_TIMESTEPS = 500_000
    print(f"Starting Stage 2 training for {TOTAL_TIMESTEPS} timesteps...")
    print("This is a harder task than Stage 1 (reach) -- expect it to take")
    print("noticeably longer. Watch 'ep_rew_mean' -- it should slowly climb")
    print("as the AI learns to touch, then grip, then lift the cube.\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save("ppo_panda_pick")
    print("\nTraining complete. Model saved as ppo_panda_pick.zip")
    print("Run evaluate_pick.py next to watch it in action.")