import numpy as np

import torch


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


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
        self.args = args
        self.episode_length = self.args.episode_length
        self.num_rollout_threads = self.args.num_rollout_threads
        self.num_agents = self.args.num_agents
        self.recurrent_N = self.args.recurrent_N
        self.hidden_size = self.args.hidden_size
        self.gamma = self.args.gamma
        self._use_gae = self.args.use_gae
        self.gae_lambda = self.args.gae_lambda
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.algorithm_name = args.algorithm_name

        obs_shape = np.array(obs_space).shape  # エージェント0の観測のshape
        share_obs_shape = np.array(
            share_obs_space
        ).shape  # エージェント0のshared観測のshape

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                self.num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )

        self.obs = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32,
        )
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.num_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)

        self.advantages = np.zeros(
            (self.episode_length, self.num_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        action_dim = len(action_space)
        action_shape = 1

        self.available_actions = np.ones(
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                num_agents,
                action_dim,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                self.episode_length,
                self.num_rollout_threads,
                num_agents,
                action_shape,
            ),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (
                self.episode_length,
                self.num_rollout_threads,
                num_agents,
                action_shape,
            ),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.num_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.num_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

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

    def compute_returns(
        self,
        next_value,
        value_normalizer=None,
    ):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.algorithm_name == "MAT" or self.algorithm_name == "MAT_DEC":
                    value_t = value_normalizer.denormalize(self.value_preds[step])
                    value_t_next = value_normalizer.denormalize(
                        self.value_preds[step + 1]
                    )
                    rewards_t = self.rewards[step]

                    delta = (
                        rewards_t
                        + self.gamma * self.masks[step + 1] * value_t_next
                        - value_t
                    )
                    gae = (
                        delta
                        + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    )
                    self.advantages[step] = gae
                    self.returns[step] = gae + value_t

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def feed_forward_generator_transformer(
        self,
        advantages,
        num_mini_batch=None,
        mini_batch_size=None,
    ):
        episode_length, num_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = num_rollout_threads * episode_length

        mini_batch_size = batch_size // num_mini_batch
        rand = torch.randperm(batch_size).numpy()

        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, *self.available_actions.shape[2:]
            )
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(
            -1, *self.action_log_probs.shape[2:]
        )
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(
                -1, *rnn_states_critic.shape[2:]
            )
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(
                    -1, *available_actions.shape[2:]
                )
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(
                -1, *active_masks.shape[2:]
            )
            old_action_log_probs_batch = action_log_probs[indices].reshape(
                -1, *action_log_probs.shape[2:]
            )
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )
