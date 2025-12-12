import copy
import random

import torch

from algorithms.hasac.algorithm.hasac_critic import Critic
from algorithms.hasac.hasac_base_trainer import BaseSACTrainer
from utils.valuenorm import ValueNorm


class CentralizedSACTrainer(BaseSACTrainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)
        total_action_dim = sum([p.action_dim for p in self.policy])
        self.critic1 = Critic(
            args,
            total_obs_dim=self.policy[0].share_obs_dim,
            total_action_dim=total_action_dim,
        )
        self.critic1.optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=self.args.critic_lr,
            eps=self.args.opti_eps,
            weight_decay=self.args.weight_decay,
        )
        self.target_critic_1 = copy.deepcopy(self.critic1)
        self.critic2 = Critic(
            args,
            total_obs_dim=self.policy[0].share_obs_dim,
            total_action_dim=total_action_dim,
        )
        self.critic2.optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=self.args.critic_lr,
            eps=self.args.opti_eps,
            weight_decay=self.args.weight_decay,
        )
        self.target_critic_2 = copy.deepcopy(self.critic2)

        self.critic1.to(self.args.device)
        self.target_critic_1.to(self.args.device)
        self.critic2.to(self.args.device)
        self.target_critic_2.to(self.args.device)
        self.use_valuenorm = getattr(self.args, "use_valuenorm", False)
        if self.use_valuenorm:
            self.target_q_valuenorm = ValueNorm(
                input_shape=self.args.episode_length,
                norm_axes=0,
                device=self.args.device,
            )

    def _train_critic(self, batch):
        target_Q = self._calculate_target_q_values(batch)
        critic_input = torch.cat([batch["share_obs"][:, :-1], self.action_env], dim=-1)
        valid_mask = (1 - batch["mask"].float()).squeeze(-1)
        mask_sum = valid_mask.sum()
        if mask_sum == 0:
            return
        q1_value = self.critic1.forward(critic_input).squeeze(-1)
        loss1 = target_Q - q1_value
        critic_loss1 = (loss1**2 * valid_mask).sum() / mask_sum
        self.critic1.optimizer.zero_grad()
        critic_loss1.backward()
        if self.args.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.critic1.parameters(), self.args.critic_max_grad_norm
            )
        self.critic1.optimizer.step()
        q2_value = self.critic2.forward(critic_input).squeeze(-1)
        loss2 = target_Q - q2_value
        critic_loss2 = (loss2**2 * valid_mask).sum() / mask_sum
        self.critic2.optimizer.zero_grad()
        critic_loss2.backward()
        if self.args.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.critic2.parameters(), self.args.critic_max_grad_norm
            )
        self.critic2.optimizer.step()

    def _train_actor(self, batch):
        # ミニエポックのループ
        for _ in range(self.args.mini_epoch_N):
            agent_indices = list(range(self.args.num_agents))
            random.shuffle(agent_indices)
            valid_mask = (1 - batch["mask"].float()).squeeze(-1)
            mask_sum = valid_mask.sum()
            if mask_sum == 0:
                continue
            for agent_id in agent_indices:
                actions = []
                for i in range(self.args.num_agents):
                    agent_obs = batch["obs"][:, :-1, i, :]
                    if i == agent_id:
                        (
                            action_env,
                            agent_log_prob,
                            agent_probs,
                        ) = self.policy[i].get_action_with_probability(agent_obs)
                    else:
                        with torch.no_grad():
                            (
                                action_env,
                                _,
                                _,
                            ) = self.policy[i].get_action_with_probability(agent_obs)
                    actions.append(action_env)

                actions_env = torch.cat(actions, dim=-1)
                input_critic = torch.cat(
                    [
                        batch["share_obs"][:, :-1],
                        actions_env,
                    ],
                    dim=-1,
                )
                Q1 = self.critic1.forward(input_critic).squeeze(-1)
                Q2 = self.critic2.forward(input_critic).squeeze(-1)
                Q = torch.min(Q1, Q2)
                if self.use_valuenorm:
                    Q = self.target_q_valuenorm.denormalize(Q, dtype="torch")
                actor_loss = (
                    self._get_alpha().detach() * agent_log_prob * valid_mask
                    - Q * valid_mask
                ).sum().sum() / valid_mask.sum()

                self.policy[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.args.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy[agent_id].actor.parameters(),
                        self.args.actor_max_grad_norm,
                    )
                self.policy[agent_id].actor_optimizer.step()
                valid_probs = agent_probs[valid_mask.bool()]
                _, alpha_loss_value = self._update_alpha(valid_probs.detach())

    def _update_target_networks(self):
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.args.tau * param.data + (1 - self.args.tau) * target_param.data
            )
        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.args.tau * param.data + (1 - self.args.tau) * target_param.data
            )

    @torch.no_grad()
    def _calculate_target_q_values(self, batch) -> torch.Tensor:
        actions_env = []
        log_probs = []
        for agent_id in range(self.args.num_agents):
            agent_obs = batch["obs"][:, 1:, agent_id, :]
            action_env, log_prob, _ = self.policy[agent_id].get_action_with_probability(
                agent_obs
            )
            actions_env.append(action_env)
            log_probs.append(log_prob)

        actions_env = torch.cat(actions_env, dim=-1)
        log_probs = torch.stack(log_probs, dim=0).sum(dim=0)

        input_critic = torch.cat(
            [
                batch["share_obs"][:, 1:],
                actions_env,
            ],
            dim=-1,
        )
        Q_1 = self.target_critic_1.forward(input_critic).squeeze(-1)
        Q_2 = self.target_critic_2.forward(input_critic).squeeze(-1)
        min_Q = torch.min(Q_1, Q_2)
        rewards = batch["rewards"].sum(dim=2).squeeze(-1)
        if self.use_valuenorm:
            min_Q_denorm = self.target_q_valuenorm.denormalize(min_Q, dtype="torch")
            next_state_value = min_Q_denorm - (self._get_alpha() * log_probs)
        else:
            next_state_value = min_Q - (self._get_alpha() * log_probs)
        done_mask = (1 - batch["mask"].float()).squeeze(-1)
        target_Q = rewards + self.args.gamma * done_mask * next_state_value
        if self.use_valuenorm:
            self.target_q_valuenorm.update(target_Q)
            target_Q = self.target_q_valuenorm.normalize(target_Q)
        return target_Q.detach()
