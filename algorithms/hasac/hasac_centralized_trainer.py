import copy
import random

import numpy as np
import torch

from algorithms.hasac.algorithm.hasac_critic import Critic
from algorithms.hasac.hasac_base_trainer import BaseSACTrainer


class CentralizedSACTrainer(BaseSACTrainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)
        # 集中型Critic（全エージェント共通）
        total_action_dim = sum([p.action_dim for p in self.policy])
        # Critic 1
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

        # Critic 2
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

        # ValueNorm for target Q
        self.use_valuenorm = getattr(self.args, "use_valuenorm", False)
        if self.use_valuenorm:
            from utils.valuenorm import ValueNorm

            # normalize over (batch, time) -> scalar normalizer for whole (B,T)
            self.target_q_valuenorm = ValueNorm(
                input_shape=1, norm_axes=2, device=self.args.device
            )

    def _train_critic(self, batch):
        if self.train_step_counter % 10 == 0:
            self.log_initial_q_landscape(batch)

        target_Q = self._calculate_target_q_values(batch)

        critic_input = torch.cat(
            [
                batch["share_obs"][:, :-1],  # obsは T+1 あるので、t=0..T-1 を使用
                self.action_env,
            ],
            dim=-1,
        )
        # ユーザーの仕様: mask=Trueは無効データ
        # maskは T あるので、そのまま使用
        valid_mask = (1 - batch["mask"].float()).squeeze(-1)
        mask_sum = valid_mask.sum()
        if mask_sum == 0:  # 全て無効データの場合は計算をスキップ
            return

        # Critic 1の更新
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

        # Critic 2の更新
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

        critic_loss = (critic_loss1 + critic_loss2) / 2
        ####
        self.critic_losses.append(critic_loss.item())
        ####

    def _train_actor(self, batch):
        # ミニエポックのループ
        for _ in range(self.args.mini_epoch_N):
            agent_indices = list(range(self.args.num_agents))
            random.shuffle(agent_indices)
            actor_losses_epoch = []
            alpha_losses_epoch = []

            # ユーザーの仕様: mask=Trueは無効データ
            valid_mask = (1 - batch["mask"].float()).squeeze(-1)
            mask_sum = valid_mask.sum()
            if mask_sum == 0:
                continue  # このエポックの学習をスキップ

            for agent_id in agent_indices:
                actions = []
                # agent_id以外のエージェントの行動を決定
                for i in range(self.args.num_agents):
                    agent_obs = batch["obs"][:, :-1, i, :]
                    if i == agent_id:
                        # 更新対象のActorは勾配を計算
                        (
                            action_env,
                            agent_log_prob,
                            agent_probs,
                        ) = self.policy[
                            i
                        ].get_action_with_probability(agent_obs)
                    else:
                        # 更新対象外のActorは勾配計算をしない
                        with torch.no_grad():
                            (
                                action_env,
                                _,
                                _,
                            ) = self.policy[
                                i
                            ].get_action_with_probability(agent_obs)
                    actions.append(action_env)

                actions_env = torch.cat(actions, dim=-1)
                input_critic = torch.cat(
                    [
                        batch["share_obs"][
                            :, :-1
                        ],  # obsは T+1 あるので、t=0..T-1 を使用
                        actions_env,
                    ],
                    dim=-1,
                )
                # 2つのCriticの出力の最小値を使用
                Q1 = self.critic1.forward(input_critic).detach().squeeze(-1)
                Q2 = self.critic2.forward(input_critic).detach().squeeze(-1)
                Q = torch.min(Q1, Q2)
                # Actor Loss: (alpha * log_pi - Q) の平均を最小化

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

                # Alphaの更新
                # マスクされた有効なステップのエントロピーのみを使用
                valid_probs = agent_probs[valid_mask.bool()]
                _, alpha_loss_value = self._update_alpha(valid_probs.detach())

                actor_losses_epoch.append(actor_loss.item())
                if alpha_loss_value is not None:
                    alpha_losses_epoch.append(alpha_loss_value)

            # 1エポック分の平均損失を記録
            if actor_losses_epoch:
                self.actor_losses.append(
                    sum(actor_losses_epoch) / len(actor_losses_epoch)
                )
            if alpha_losses_epoch:
                self.alpha_losses.append(
                    sum(alpha_losses_epoch) / len(alpha_losses_epoch)
                )

    def _update_target_networks(self):
        # Critic 1
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.args.tau * param.data + (1 - self.args.tau) * target_param.data
            )
        # Critic 2
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
            # obsは T+1 あるので、t=1..T を使用
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
                batch["share_obs"][:, 1:],  # obsは T+1 あるので、t=1..T を使用
                actions_env,
            ],
            dim=-1,
        )
        Q_1 = self.target_critic_1.forward(input_critic).squeeze(-1)
        Q_2 = self.target_critic_2.forward(input_critic).squeeze(-1)
        min_Q = torch.min(Q_1, Q_2)
        rewards = batch["rewards"].sum(dim=2).squeeze(-1)
        ####
        # 1エピソードあたりの合計報酬を計算し、バッチ内のエピソードで平均を取る
        # (B, T, N, 1) -> sum over T, N, 1 -> (B,) -> mean -> scalar
        episode_rewards = batch["rewards"].sum(dim=(1, 2, 3))
        self.rewards.append(episode_rewards.mean().item())
        if self.use_valuenorm:
            # Use running mean/var to denormalize in-torch (avoid numpy round-trip)
            mean, var = self.target_q_valuenorm.running_mean_var()
            sigma = torch.sqrt(var).to(min_Q.device).to(min_Q.dtype)
            min_Q_denorm = min_Q * sigma + mean.to(min_Q.device).to(min_Q.dtype)
            next_state_value = min_Q_denorm - (self._get_alpha() * log_probs)
        else:
            # critics are raw-scale, safe to subtract directly
            next_state_value = min_Q - (self._get_alpha() * log_probs)

        # 完了状態（mask=True）では次の状態の価値は0
        # maskは T あるので、そのまま使用
        done_mask = (1 - batch["mask"].float()).squeeze(-1)

        target_Q = rewards + self.args.gamma * done_mask * next_state_value

        if self.use_valuenorm:
            # Ensure shape is as expected for debugging
            # (B, T)
            # print(target_Q.shape)
            # Update running stats with raw-scale targets
            self.target_q_valuenorm.update(target_Q)
            # Normalize targets for critic loss
            target_Q = self.target_q_valuenorm.normalize(target_Q)
        return target_Q.detach()

    def log_initial_q_landscape(self, batch):
        """
        各エージェントの初期状態での各行動(one-hot)ごとのQ値をlog出力
        """
        with torch.no_grad():
            initial_share_obs = batch["share_obs"][:, 0, :]
            initial_actions = []
            for i in range(self.args.num_agents):
                initial_obs_agent = batch["obs"][:, 0, i, :]
                _, action_env = self.policy[i].get_action(
                    initial_obs_agent, deterministic=False
                )
                initial_actions.append(action_env)

            for agent_id in range(self.args.num_agents):
                act_dim = self.policy[agent_id].action_dim
                batch_size = initial_share_obs.shape[0]
                other_actions = [
                    initial_actions[i].clone()
                    for i in range(self.args.num_agents)
                    if i != agent_id
                ]
                q_per_action = []
                for act_i in range(act_dim):
                    one_hot_act = torch.zeros(
                        batch_size, act_dim, device=self.args.device
                    )
                    one_hot_act[:, act_i] = 1
                    temp_actions = []
                    other_idx = 0
                    for i in range(self.args.num_agents):
                        if i == agent_id:
                            temp_actions.append(one_hot_act)
                        else:
                            temp_actions.append(other_actions[other_idx])
                            other_idx += 1
                    temp_actions_env = torch.cat(temp_actions, dim=-1)
                    critic_input = torch.cat(
                        [initial_share_obs, temp_actions_env], dim=-1
                    )
                    q_val = self.critic1.forward(critic_input)
                    q_per_action.append(q_val.mean().item())
                print(
                    f"[DEBUG] Agent {agent_id} Q-values per action (init): {np.round(q_per_action, 4)}"
                )
