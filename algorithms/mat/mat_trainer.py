from utils.valuenorm import ValueNorm
import numpy as np
from algorithms.utils.util import check

import torch
import torch.nn as nn
from utils.util import get_grad_norm, huber_loss, mse_loss


class MATTrainer:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model,
    policy, and env information.
    :param policy: (RMAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args,
        policy,
    ):
        self.args = args
        self.policy = policy
        self.num_agents = self.args.num_agents
        self.device = self.args.device
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # PPO parameters
        self.clip_param = self.args.clip_param
        self.ppo_epoch = self.args.ppo_epoch
        self.num_mini_batch = self.args.num_mini_batch
        self.value_loss_coef = self.args.value_loss_coef
        self.value_loss_coef = self.args.value_loss_coef
        self.entropy_coef = self.args.entropy_coef
        self.max_grad_norm = self.args.max_grad_norm
        self.huber_delta = self.args.huber_delta

        self._use_recurrent_policy = self.args.use_recurrent_policy
        self._use_max_grad_norm = self.args.use_max_grad_norm
        self._use_clipped_value_loss = self.args.use_clipped_value_loss
        self._use_huber_loss = self.args.use_huber_loss
        self._use_valuenorm = self.args.use_valuenorm
        self._use_value_active_masks = self.args.use_value_active_masks
        self._use_policy_active_masks = self.args.use_policy_active_masks

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(
                input_shape=1,
                device=self.device,
            )
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch
    ):

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        (
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
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        loss = (
            policy_loss
            - dist_entropy * self.entropy_coef
            + value_loss * self.value_loss_coef
        )

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.transformer.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = get_grad_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return (
            value_loss,
            grad_norm,
            policy_loss,
            dist_entropy,
            grad_norm,
            imp_weights,
        )

    def train(self, buffer):
        """Minibatch GD updateで学習する"""
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)

        train_infos = {}

        train_infos["value_loss"] = 0
        train_infos["policy_loss"] = 0
        train_infos["dist_entropy"] = 0
        train_infos["actor_grad_norm"] = 0
        train_infos["critic_grad_norm"] = 0
        train_infos["ratio"] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(
                advantages,
                self.num_mini_batch,
            )
            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample)
                train_infos["value_loss"] += value_loss.item()
                train_infos["policy_loss"] += policy_loss.item()
                train_infos["dist_entropy"] += dist_entropy.item()
                train_infos["actor_grad_norm"] += actor_grad_norm
                train_infos["critic_grad_norm"] += critic_grad_norm
                train_infos["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_infos.keys():
            train_infos[k] /= num_updates

        return train_infos

    def prep_rollout(
        self,
    ):
        self.policy.eval()

    def prep_training(
        self,
    ):
        self.policy.train()
