import copy
import random

import torch

from algorithms.hasac.algorithm.hasac_critic import Critic
from algorithms.hasac.hasac_base_trainer import BaseSACTrainer
from utils.valuenorm import ValueNorm


class IndependentSACTrainer(BaseSACTrainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)
        self.use_valuenorm = self.args.use_valuenorm
        if self.use_valuenorm:
            self.target_q_valuenorm = [
                ValueNorm(
                    input_shape=self.args.episode_length,
                    norm_axes=0,
                    device=self.args.device,
                )
                for _ in range(self.args.num_agents)
            ]
        self.critic1 = []
        self.critic2 = []
        for i in range(self.args.num_agents):
            critic1 = Critic(
                args,
                total_obs_dim=self.policy[i].obs_dim,
                total_action_dim=self.policy[i].action_dim,
            )
            critic1.optimizer = torch.optim.Adam(
                critic1.parameters(),
                lr=self.args.critic_lr,
                eps=self.args.opti_eps,
                weight_decay=self.args.weight_decay,
            )
            self.critic1.append(critic1)
            critic2 = Critic(
                args,
                total_obs_dim=self.policy[i].obs_dim,
                total_action_dim=self.policy[i].action_dim,
            )
            critic2.optimizer = torch.optim.Adam(
                critic2.parameters(),
                lr=self.args.critic_lr,
                eps=self.args.opti_eps,
                weight_decay=self.args.weight_decay,
            )
            self.critic2.append(critic2)

        self.target_critic_1 = copy.deepcopy(self.critic1)
        self.target_critic_2 = copy.deepcopy(self.critic2)

        for i in range(self.args.num_agents):
            self.critic1[i].to(self.args.device)
            self.critic2[i].to(self.args.device)
            self.target_critic_1[i].to(self.args.device)
            self.target_critic_2[i].to(self.args.device)

    def _train_critic(self, batch):
        valid_mask = (1 - batch["mask"].float()).squeeze(-1)
        mask_sum = valid_mask.sum()
        if mask_sum == 0:
            return
        for agent_id in range(self.args.num_agents):
            target_Q = self._calculate_target_q_values(batch, agent_id)
            agent_action_env = self.action_env.view(
                self.args.batch_size,
                self.args.episode_length,
                self.args.num_agents,
                -1,
            )[:, :, agent_id, :]
            critic_input = torch.cat(
                [batch["obs"][:, :-1, agent_id, :], agent_action_env], dim=-1
            )
            if self.use_valuenorm:
                self.target_q_valuenorm[agent_id].update(target_Q)
                target_Q_norm = self.target_q_valuenorm[agent_id].normalize(target_Q)
            else:
                target_Q_norm = target_Q
            q1_value = self.critic1[agent_id].forward(critic_input).squeeze(-1)
            loss1 = target_Q_norm - q1_value
            critic_loss1 = (loss1**2 * valid_mask).sum() / mask_sum
            self.critic1[agent_id].optimizer.zero_grad()
            critic_loss1.backward()
            if self.args.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.critic1[agent_id].parameters(), self.args.critic_max_grad_norm
                )
            self.critic1[agent_id].optimizer.step()
            q2_value = self.critic2[agent_id].forward(critic_input).squeeze(-1)
            loss2 = target_Q_norm - q2_value
            critic_loss2 = (loss2**2 * valid_mask).sum() / mask_sum
            self.critic2[agent_id].optimizer.zero_grad()
            critic_loss2.backward()
            if self.args.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.critic2[agent_id].parameters(), self.args.critic_max_grad_norm
                )
            self.critic2[agent_id].optimizer.step()

    def _train_actor(self, batch):
        for _ in range(self.args.mini_epoch_N):
            agent_indices = list(range(self.args.num_agents))
            random.shuffle(agent_indices)
            valid_mask = (1 - batch["mask"].float()).squeeze(-1)
            mask_sum = valid_mask.sum()
            if mask_sum == 0:
                continue
            for agent_id in agent_indices:
                agent_obs = batch["obs"][:, :-1, agent_id, :]
                action_env, log_prob, probs = self.policy[
                    agent_id
                ].get_action_with_probability(agent_obs)

                input_critic = torch.cat([agent_obs, action_env], dim=-1)

                Q1 = self.critic1[agent_id].forward(input_critic).squeeze(-1)
                Q2 = self.critic2[agent_id].forward(input_critic).squeeze(-1)
                Q = torch.min(Q1, Q2)
                if self.use_valuenorm:
                    Q = self.target_q_valuenorm[agent_id].denormalize(Q, dtype="torch")

                actor_loss = (
                    self._get_alpha().detach() * log_prob * valid_mask - Q * valid_mask
                ).sum() / valid_mask.sum()

                self.policy[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.args.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy[agent_id].actor.parameters(),
                        self.args.actor_max_grad_norm,
                    )
                self.policy[agent_id].actor_optimizer.step()

                valid_probs = probs[valid_mask.bool()]
                _, alpha_loss_value = self._update_alpha(valid_probs.detach())

    @torch.no_grad()
    def _calculate_target_q_values(self, batch, agent_id) -> torch.Tensor:
        agent_obs_next = batch["obs"][:, 1:, agent_id, :]
        action_env_next, log_prob_next, _ = self.policy[
            agent_id
        ].get_action_with_probability(agent_obs_next)
        input_critic = torch.cat([agent_obs_next, action_env_next], dim=-1)
        Q_1 = self.target_critic_1[agent_id].forward(input_critic).squeeze(-1)
        Q_2 = self.target_critic_2[agent_id].forward(input_critic).squeeze(-1)
        min_Q = torch.min(Q_1, Q_2)
        rewards = batch["rewards"][:, :, agent_id, 0]
        if self.use_valuenorm:
            min_Q_denorm = self.target_q_valuenorm[agent_id].denormalize(
                min_Q, dtype="torch"
            )
            next_state_value = min_Q_denorm - (self._get_alpha() * log_prob_next)
        else:
            next_state_value = min_Q - (self._get_alpha() * log_prob_next)
        done_mask = (1 - batch["mask"].float()).squeeze(-1)
        target_Q = rewards + self.args.gamma * done_mask * next_state_value
        return target_Q.detach()

    def _update_target_networks(self):
        for agent_id in range(self.args.num_agents):
            for target_param, param in zip(
                self.target_critic_1[agent_id].parameters(),
                self.critic1[agent_id].parameters(),
            ):
                target_param.data.copy_(
                    self.args.tau * param.data + (1 - self.args.tau) * target_param.data
                )
            for target_param, param in zip(
                self.target_critic_2[agent_id].parameters(),
                self.critic2[agent_id].parameters(),
            ):
                target_param.data.copy_(
                    self.args.tau * param.data + (1 - self.args.tau) * target_param.data
                )
