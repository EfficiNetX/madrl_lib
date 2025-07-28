import numpy as np


class MultiAgentEnv:
    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None,
        post_step_callback=None,
        shared_viewer=True,
        discrete_action=True,
    ):
        self.world = world
        self.num_agents = len(world.agents)

        self.reset_callback = reset_callback
        self.observation_callback = observation_callback

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []

        self.observation_space = [
            [-1 for x in range(self.num_agents)] + [-1, -1] * self.num_agents
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            ([-1 for x in range(self.num_agents)] + [-1, -1] * self.num_agents)
            * self.num_agents
            for _ in range(self.num_agents)
        ]
        self.action_space = [[-1] for _ in range(self.num_agents)]

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)

        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        for i, agent in enumerate(self.agents):
            agent_id_feats = np.zeros(
                self.num_agents,
                dtype=np.float32,
            )
            agent_id_feats[i] = 1.0
            obs_i = np.concatenate([agent_id_feats, self._get_obs(agent)])
            obs_n.append(obs_i)
        return obs_n

    def step(self, actions):
        pass

    def _get_info(self, agent):
        pass

    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    def _get_done(self, agent):
        pass

    def _get_reward(self, agent):
        pass
