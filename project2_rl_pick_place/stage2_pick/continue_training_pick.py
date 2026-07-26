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

    print("Loading existing model: ppo_panda_pick.zip ...")
    model = PPO.load("ppo_panda_pick", env=env)

   
    ADDITIONAL_TIMESTEPS = 500_000
    print(f"Continuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")
    print("It's already succeeding sometimes -- this run should push the")
    print("success rate higher. Watch 'ep_len_mean' -- it should start")
    print("dropping noticeably below 300 as successes become more frequent.\n")

    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)

    model.save("ppo_panda_pick")
    print("\nDone. Model updated: ppo_panda_pick.zip")
    print("Run evaluate_pick.py again to check progress.")