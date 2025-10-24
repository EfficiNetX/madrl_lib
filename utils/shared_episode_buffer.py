import numpy as np


class EpisodeReplayBuffer:
    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.buffer_size = args.buffer_size
        self.episode_length = args.episode_length
        self.num_rollout_threads = args.num_rollout_threads
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.buffer_index = 0
        self.episodes_in_buffer = 0  # バッファに保存されているエピソード数
        self.buffer_size = args.buffer_size
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
        self.avail_actions = np.zeros(
            (
                self.buffer_size,
                self.episode_length + 1,
                self.num_agents,
                len(action_space),
            ),
            dtype=bool,
        )

    def _assign_buffer(
        self,
        idx: slice,
        share_obs: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        mask: np.ndarray,
        avail_actions: np.ndarray,
        offset: int = 0,
    ):
        length = idx.stop - idx.start
        sl = slice(offset, offset + length)
        self.share_obs[idx] = share_obs[sl]
        self.obs[idx] = obs[sl]
        self.actions[idx] = actions[sl]
        self.rewards[idx] = rewards[sl]
        self.dones[idx] = dones[sl]
        self.mask[idx] = mask[sl]
        self.avail_actions[idx] = avail_actions[sl]

    def insert(
        self,
        share_obs: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        mask: np.ndarray,
        avail_actions: np.ndarray,
    ):
        """
        他のBufferクラスと統一したインターフェース

        Args:
            share_obs: (num_envs, episode_length + 1, share_obs_dim)
            obs: (num_envs, episode_length + 1, num_agents, obs_dim)
            actions: (num_envs, episode_length, num_agents, 1)
            rewards: (num_envs, episode_length, num_agents, 1)
            dones: (num_envs, episode_length, num_agents)
            mask: (num_envs, episode_length)
            avail_actions: (num_envs, episode_length + 1, num_agents, action_dim)
        """
        # 直接numpy配列に代入（他のBufferと同じパターン）
        if self.buffer_index + self.num_rollout_threads <= self.buffer_size:
            idx = slice(self.buffer_index, self.buffer_index + self.num_rollout_threads)
            self._assign_buffer(
                idx,
                share_obs,
                obs,
                actions,
                rewards,
                dones,
                mask,
                avail_actions,
                offset=0,
            )
            self.buffer_index += self.num_rollout_threads
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
        else:
            # バッファがオーバーフローする場合
            overflow = self.buffer_index + self.num_rollout_threads - self.buffer_size
            # 前半部分
            idx_first = slice(self.buffer_index, self.buffer_size)
            num_first = self.buffer_size - self.buffer_index
            self._assign_buffer(
                idx_first,
                share_obs,
                obs,
                actions,
                rewards,
                dones,
                mask,
                avail_actions,
                offset=0,
            )

            # 後半部分（バッファの先頭に戻る）
            idx_second = slice(0, overflow)
            self._assign_buffer(
                idx_second,
                share_obs,
                obs,
                actions,
                rewards,
                dones,
                mask,
                avail_actions,
                offset=num_first,
            )
            self.buffer_index = overflow

    def can_sample(self, batch_size: int) -> bool:
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size: int) -> dict:
        indices = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        batch = dict(
            share_obs=self.share_obs[indices],
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            mask=self.mask[indices],
            avail_actions=self.avail_actions[indices],
        )
        return batch
