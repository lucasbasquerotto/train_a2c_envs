import torch
from gymnasium.vector.vector_env import VectorEnv

class Shapes():
    def __init__(self, envs: VectorEnv):
        self.obs_shape = envs.single_observation_space.shape[0]
        self.n_actions = envs.single_action_space.n

class Settings():
    def __init__(self):
        # environment hyperparams
        self.n_envs = 10
        self.n_updates = 1000
        self.n_steps_per_update = 128
        self.randomize_domain = False

        # agent hyperparams
        self.gamma = 0.999
        self.lam = 0.95  # hyperparameter for GAE
        self.ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic

        # set the device
        use_cuda = False

        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        self.device = device

        self.save_weights = True
        self.load_weights = True

        base_dir = "data/weights"
        self.base_dir = base_dir

        self.actor_weights_path = f"{base_dir}/actor_weights.h5"
        self.critic_weights_path = f"{base_dir}/critic_weights.h5"

    def shapes(self, envs: VectorEnv) -> Shapes:
        return Shapes(envs)