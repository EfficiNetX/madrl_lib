import numpy as np
import random
import torch


class EpisodeReplayBuffer:
    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.buffer_size = args.qmix_buffer_size
        self.episode_length = args.episode_length
        self.num_rollout_threads = args.num_rollout_threads
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.buffer_index = 0
        self.episodes_in_buffer = 0  # バッファに保存されているエピソード数
        self.buffer_size = args.qmix_buffer_size
        self.batch_size = args.qmix_batch_size
        self.share_obs_dim = len(share_obs_space)
        self.obs_dim = len(obs_space)

        # バッファ本体
        self.share_obs = np.zeros(
            (
                self.buffer_size,
                self.episode_length + 1,
                self.share_obs_dim,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (
                self.buffer_size,
                self.episode_length + 1,
                self.num_agents,
                self.obs_dim,
            ),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (
                self.buffer_size,
                self.episode_length,
                self.num_agents,
                1,
            ),
            dtype=np.int64,
        )
        self.rewards = np.zeros(
            (
                self.buffer_size,
                self.episode_length,
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )
        self.dones = np.zeros(
            (
                self.buffer_size,
                self.episode_length,
                self.num_agents,
            ),
            dtype=bool,
        )
        self.mask = np.zeros(
            (
                self.buffer_size,
                self.episode_length,
            ),
            dtype=bool,
        )

    def _assign_buffer(self, idx, episodes_data, offset=0):
        self.share_obs[idx] = episodes_data["share_obs"][offset : offset + idx.stop - idx.start]
        self.obs[idx] = episodes_data["obs"][offset : offset + idx.stop - idx.start]
        self.actions[idx] = episodes_data["actions"][offset : offset + idx.stop - idx.start]
        self.rewards[idx] = episodes_data["rewards"][offset : offset + idx.stop - idx.start]
        self.dones[idx] = episodes_data["dones"][offset : offset + idx.stop - idx.start]
        self.mask[idx] = episodes_data["mask"][offset : offset + idx.stop - idx.start]

    def insert(self, episodes_data):
        """
        episodes_data: dict
            {
                "share_obs": (num_envs, episode_length + 1, num_agents, share_obs_dim)
                "obs": (num_envs, episode_length + 1, num_agents, obs_dim)
                "actions": (num_envs, episode_length, num_agents, 1)
                "rewards": (num_envs, episode_length, num_agents, 1)
                "dones": (num_envs, episode_length, num_agents)
                "mask": (num_envs, episode_length)
            }
        """
        if self.buffer_index + self.num_rollout_threads <= self.buffer_size:
            idx = slice(self.buffer_index, self.buffer_index + self.num_rollout_threads)
            self._assign_buffer(idx, episodes_data)
            self.buffer_index += self.num_rollout_threads
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
        else:
            overflow = self.buffer_index + self.num_rollout_threads - self.buffer_size
            idx_first = slice(self.buffer_index, self.buffer_size)
            idx_second = slice(0, overflow)
            self._assign_buffer(idx_first, episodes_data, offset=0)
            self._assign_buffer(
                idx_second,
                episodes_data,
                offset=self.buffer_size - self.buffer_index,
            )
            self.buffer_index = overflow

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= self.batch_size

    def sample(self, batch_size):
        indices = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        batch = dict(
            share_obs=self.share_obs[indices],
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            mask=self.mask[indices],
        )
        return batch
