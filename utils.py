import os
import torch
from settings import Settings
from a2c import A2C

class Utils():
    def __init__(self, settings: Settings):
        self.settings = settings

    def _mkdir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def save_weights(self, agent: A2C) -> None:
        settings = self.settings
        self._mkdir(settings.base_dir)
        torch.save(agent.actor.state_dict(), settings.actor_weights_path)
        torch.save(agent.critic.state_dict(), settings.critic_weights_path)

    def load_weights(self, agent: A2C) -> None:
        settings = self.settings
        self._mkdir(settings.base_dir)
        agent.actor.load_state_dict(torch.load(settings.actor_weights_path))
        agent.critic.load_state_dict(torch.load(settings.critic_weights_path))
        agent.actor.eval()
        agent.critic.eval()
