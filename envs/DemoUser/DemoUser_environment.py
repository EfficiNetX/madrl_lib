import numpy as np
import torch


class DemoUserMultiAgentEnv:
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
        self.world_length = self.world.world_length
        self.current_step = 0

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []

        # 各エージェントが同じ報酬を受け取るかどうか
        self.shared_reward = False

        self.observation_space = [
            [-1 for x in range(self.num_agents)] + [-1, -1, -1, -1]
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            ([-1 for x in range(self.num_agents)] + [-1, -1, -1, -1]) * self.num_agents
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            [-1 for _ in range(5)] for _ in range(self.num_agents)
        ]  # 縦・横・とどまる

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

    def step(self, actions_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        self.agents = self.world.agents
        # set action of each agent
        for i, agent in enumerate(self.agents):
            self._set_action(
                action=torch.argmax(torch.tensor(actions_n[i])).item(),
                agent=agent,
                action_space=self.action_space[i],
            )
        # advance world state
        self.world.step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(
                np.concatenate(
                    [
                        np.array([1 if j == i else 0 for j in range(self.num_agents)]),
                        self._get_obs(agent),
                    ]
                )
            )
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.num_agents
        else:
            reward_n = [[r] for r in reward_n]
        return obs_n, reward_n, done_n

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space):
        agent.action.move = np.zeros(2)
        if action == 0:
            agent.action.move[0] = -1
        elif action == 1:
            agent.action.move[0] = 1
        elif action == 2:
            agent.action.move[1] = -1
        elif action == 3:
            agent.action.move[1] = 1

    def _get_info(self, agent):
        pass

    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    def _get_done(self, agent):
        if self.current_step >= self.world_length:
            return True
        else:
            return False

    def _get_reward(self, agent):
        return self.reward_callback(agent, self.world)
