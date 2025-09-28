import numpy as np
import random
import torch


class EpisodeReplayBuffer:
    # 1エピソード単位で経験を保存するリプレイバッファ
    """
    保存するデータの形式:
    episode = ()

    """

    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.buffer_size = args.qmix_buffer_size
        self.buffer = []
        self.position = 0

    def add(self, episode):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(episode)
        else:
            self.buffer[self.position] = episode
            self.position = (self.position + 1) % self.buffer_size

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
