from gymnasium.utils.play import play

from env import make_env

env = make_env(randomize_domain=False, render_mode="rgb_array")

play(env, keys_to_action={'w': 2, 'a': 1, 'd': 3}, noop=0)
