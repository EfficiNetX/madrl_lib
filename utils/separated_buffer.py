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
            (self.episode_length + 1, self.num_rollout_threads, share_obs_shape),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, obs_shape),
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

        self.values_preds = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, 1),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.episode_length, self.num_rollout_threads, action_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.num_rollout_threads, action_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.num_rollout_threads, 1), dtype=np.float32
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.num_rollout_threads, 1), dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            self.values_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.episode_length)):
                if self._use_popart or self._use_valuenorm:
                    delta = (
                        self.rewards[step]
                        + self.gamma
                        * value_normalizer.denormalize(self.values_preds[step + 1])
                        * self.masks[step + 1]
                        - value_normalizer.denormalize(self.values_preds[step])
                    )
                    gae = (
                        delta
                        + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    )
                    self.returns[step] = gae + value_normalizer.denormalize(
                        self.values_preds[step]
                    )

    def insert(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        self.step = (self.step + 1) % self.episode_length
