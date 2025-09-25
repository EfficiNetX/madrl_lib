import numpy as np
import torch


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


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
        action_shape = 1

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
            (self.episode_length, self.num_rollout_threads, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.num_rollout_threads, 1), dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        # if self.available_actions is not None:
        #     self.available_actions[0] = self.available_actions[-1].copy()

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

    def update_factor(self, factor):
        self.factor = factor.copy()

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
        self.values_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
            obs = (
                self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
            )
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.values_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        if self.factor is not None:
            factor = _cast(self.factor)
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states.shape[2:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        # if self.available_actions is not None:
        #     available_actions = _cast(self.available_actions[:-1])

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
            factor_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                # if self.available_actions is not None:
                #     available_actions_batch.append(
                #         available_actions[ind : ind + data_chunk_length]
                #     )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind : ind + data_chunk_length])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = np.stack(share_obs_batch)
            obs_batch = np.stack(obs_batch)

            actions_batch = np.stack(actions_batch)
            # if self.available_actions is not None:
            #     available_actions_batch = np.stack(available_actions_batch)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[2:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            # if self.available_actions is not None:
            #     available_actions_batch = _flatten(L, N, available_actions_batch)
            # else:
            available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
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
                factor_batch,
            )
