from reach_env import PandaReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    def _init():
        env = PandaReachEnv(render=False)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    NUM_PARALLEL_ENVS = 4
    env = SubprocVecEnv([make_env() for _ in range(NUM_PARALLEL_ENVS)])

    print("Loading existing model: ppo_panda_reach.zip ...")
    model = PPO.load("ppo_panda_reach", env=env)

    
    ADDITIONAL_TIMESTEPS = 300_000
    print(f"Continuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")
    print("Watch 'ep_rew_mean' -- it should keep climbing (getting less negative).\n")

    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)

    model.save("ppo_panda_reach")
    print("\nDone. Model updated: ppo_panda_reach.zip")
    print("Run evaluate_reach.py again to see if it's improved.")