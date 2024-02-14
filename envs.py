import gymnasium as gym
from env import make_env

def make_envs(n_envs: int, randomize_domain=False, render_mode: str | None=None):
    return gym.vector.AsyncVectorEnv(
        [
            lambda: make_env(
                randomize_domain=randomize_domain,
                render_mode=render_mode)
            for _ in range(n_envs)
        ]
    )