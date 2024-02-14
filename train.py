import gymnasium as gym
from tqdm import tqdm
import torch

from a2c import A2C
from settings import Settings
from envs import make_envs
from utils import Utils

class TrainResult():
    def __init__(self, 
            settings: Settings,
            agent: A2C, 
            envs_wrapper: gym.wrappers.RecordEpisodeStatistics, 
            critic_losses: list[float], 
            actor_losses: list[float], 
            entropies: list[float]):
        self.settings = settings
        self.agent = agent
        self.envs_wrapper = envs_wrapper
        self.critic_losses = critic_losses
        self.actor_losses = actor_losses
        self.entropies = entropies

def train(settings: Settings) -> TrainResult:
    utils = Utils(settings=settings)

    envs = make_envs(n_envs=settings.n_envs, randomize_domain=settings.randomize_domain)

    shapes = settings.shapes(envs)
    obs_shape = shapes.obs_shape
    action_shape = shapes.n_actions

    device = settings.device
    critic_lr = settings.critic_lr
    actor_lr = settings.actor_lr
    n_envs = settings.n_envs

    n_updates = settings.n_updates
    n_steps_per_update = settings.n_steps_per_update
    gamma = settings.gamma
    lam = settings.lam
    ent_coef = settings.ent_coef

    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    if settings.load_weights:
        utils.load_weights(agent)

    # create a wrapper environment to save episode returns and episode lengths
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []

    # use tqdm to get a progress bar for training
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode (sample phase)
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)

        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                actions.cpu().numpy()
            )

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

    result = TrainResult(
        settings=settings, 
        agent=agent, 
        envs_wrapper=envs_wrapper, 
        critic_losses=critic_losses, 
        actor_losses=actor_losses, 
        entropies=entropies)
    
    if settings.save_weights:
        utils.save_weights(agent)

    return result