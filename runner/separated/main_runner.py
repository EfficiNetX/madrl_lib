from runner.separated.base_runner import BaseRunner
from itertools import chain
import numpy as np


class UserEnvRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()

    def warmup(self):
        # reset the env
        obs = self.envs.reset()
        share_obs = []

        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
