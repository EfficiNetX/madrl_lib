import numpy as np

import torch


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


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
            (
                self.episode_length + 1,
                self.num_rollout_threads,
                num_agents,
                1,
            ),
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
                if self.algorithm_name == "MAT":
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
                else:
                    delta = (
                        self.rewards[step]
                        + self.gamma
                        * value_normalizer.denormalize(self.value_preds[step + 1])
                        * self.masks[step + 1]
                        - value_normalizer.denormalize(self.value_preds[step])
                    )
                    gae = (
                        delta
                        + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    )
                    self.returns[step] = gae + value_normalizer.denormalize(
                        self.value_preds[step]
                    )

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

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        if len(self.share_obs.shape) > 4:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.share_obs.shape[3:])
            )
            obs = (
                self.obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.obs.shape[3:])
            )
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states.shape[3:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[3:])
        )

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind : ind + data_chunk_length]
                    )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

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
