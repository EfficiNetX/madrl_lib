import numpy as np


class ReplayBuffer(object):
    """訓練データをストアするバッファ"""

    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = args.num_agents

        obs_shape = np.array(obs_space).shape  # エージェント0の観測のshape
        share_obs_shape = np.array(
            share_obs_space
        ).shape  # エージェント0のshared観測のshape

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )

        self.obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *obs_shape,
            ),
            dtype=np.float32,
        )
