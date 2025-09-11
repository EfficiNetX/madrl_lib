import numpy as np


class SeparatedReplayBuffer(object):
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.args = args

        self.episode_length = self.args.episode_length
        self.num_rollout_threads = self.args.num_rollout_threads
        self.num_agents = self.args.num_agents
        self.recurrent_N = self.args.recurrent_N
        self.rnn_hidden_size = self.args.hidden_size
        self.gamma = self.args.gamma
        self._use_gae = self.args.use_gae
        self.gae_lambda = self.args.gae_lambda
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.algorithm_name = args.algorithm_name

        obs_shape = len(obs_space)
        share_obs_shape = len(share_obs_space)
        action_shape = len(action_space)

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                share_obs_shape,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                obs_shape,
            ),
            dtype=np.float32,
        )

        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                self.recurrent_N,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(
            self.rnn_states,
        )

        self.values_pred = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, 1), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, 1), dtype=np.float32
        )
