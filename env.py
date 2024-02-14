import gymnasium as gym
import numpy as np

def make_env(randomize_domain=False, render_mode: str | None=None):
    # create a new sample environment to get new random parameters
    if randomize_domain:
        env = gym.make(
            "LunarLander-v2",
            render_mode=render_mode,
            gravity=np.clip(
                np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
            ),
            enable_wind=np.random.choice([True, False]),
            wind_power=np.clip(
                np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
            ),
            turbulence_power=np.clip(
                np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
            ),
            max_episode_steps=500,
        )
        return env
    else:
        env = gym.make("LunarLander-v2", render_mode=render_mode, max_episode_steps=500)
        return env