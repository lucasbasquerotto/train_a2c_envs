import torch

from env import make_env
from a2c import A2C

def showcase(n_showcase_episodes: int, randomize_domain: bool, agent: A2C):
    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")
        
        env = make_env(randomize_domain=randomize_domain, render_mode="human")

        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                # if state is a np.array, state[None, :] is equivalent 
                # to state.reshape(1, -1), or alternatively, np.array([state])
                action, _, _, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(action.item())

            # update if the environment is done
            done = terminated or truncated

    env.close()