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

    print("Loading existing model: ppo_panda_pick_vision.zip ...")
    model = PPO.load("ppo_panda_pick_vision", env=env)

    ADDITIONAL_TIMESTEPS = 1_50_000
    print(f"Continuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")
    print("Watch 'ep_len_mean' -- it should start dropping below 300 once")
    print("it begins actually succeeding, same pattern as the ground-truth")
    print("pick task did.\n")

    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)

    model.save("ppo_panda_pick_vision")
    print("\nDone. Model updated: ppo_panda_pick_vision.zip")
    print("Run evaluate_pick_vision.py again to check progress.")