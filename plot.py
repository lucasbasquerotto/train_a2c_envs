import numpy as np
import matplotlib.pyplot as plt

from train import TrainResult

def plot(train_result: TrainResult):
    settings = train_result.settings
    agent = train_result.agent
    envs_wrapper = train_result.envs_wrapper
    critic_losses = train_result.critic_losses
    actor_losses = train_result.actor_losses
    entropies = train_result.entropies

    n_envs = settings.n_envs
    n_steps_per_update = settings.n_steps_per_update
    randomize_domain = settings.randomize_domain

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment \n \
                (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(
            np.array(envs_wrapper.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")


    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(
            np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")


    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()