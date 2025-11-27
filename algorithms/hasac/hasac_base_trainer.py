from datetime import datetime

import numpy as np
import torch


class BaseSACTrainer:
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        # alphaの設定
        ####
        self.log_file = "log.txt"
        self.log_interval = 50
        self.train_step_counter = 0
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.rewards = []
        self.alphas = []
        self.entropies = []
        self.initial_q_values = []

        if self.args.auto_alpha:
            # log_alphaを初期化 (log(initial_alpha))
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(
                    np.log(self.args.initial_alpha),
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.args.device,
                )
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.args.alpha_lr,
                eps=self.args.opti_eps,
                weight_decay=self.args.weight_decay,
            )

            if self.args.action_type == "discrete":
                action_dim = self.policy[0].action_dim
                self.target_entropy = (
                    -np.log(1.0 / action_dim) * self.args.target_entropy_ratio
                )
            else:
                # 連続行動の場合: -action_dim (1エージェント分)
                self.target_entropy = (
                    -self.policy[0].action_dim * self.args.target_entropy_ratio
                )
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None

    def train(self, episode_samples):
        """共通のtrainフロー"""
        ####
        self.train_step_counter += 1  # 【追加】ステップカウンタをインクリメント
        ####
        batch = {
            "share_obs": torch.tensor(
                episode_samples["share_obs"], device=self.args.device
            ),
            "obs": torch.tensor(episode_samples["obs"], device=self.args.device),
            "actions": torch.tensor(
                episode_samples["actions"], device=self.args.device
            ),
            "rewards": torch.tensor(
                episode_samples["rewards"], device=self.args.device
            ),
            "dones": torch.tensor(episode_samples["dones"], device=self.args.device),
            "mask": torch.tensor(episode_samples["mask"], device=self.args.device),
            "avail_actions": torch.tensor(
                episode_samples["avail_actions"], device=self.args.device
            ),
        }
        self.action_env = self._create_action_env(batch)
        self._train_critic(batch)
        self._train_actor(batch)
        self._update_target_networks()

        self.alphas.append(self._get_alpha().item())
        if self.train_step_counter % self.log_interval == 0:
            self._log_stats()

    def _log_stats(self):
        avg_critic_loss = np.mean(self.critic_losses)
        avg_actor_loss = np.mean(self.actor_losses)
        avg_alpha_loss = np.mean(self.alpha_losses)
        avg_reward = np.mean(self.rewards)
        avg_alpha = np.mean(self.alphas)
        avg_entropy = np.mean(self.entropies)
        avg_initial_q = (
            np.mean(self.initial_q_values) if self.initial_q_values else "N/A"
        )

        log_str = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f" Step: {self.train_step_counter}\n"
            f"\tAvg Critic Loss: {avg_critic_loss:.6f}\n"
            f"\tAvg Actor Loss: {avg_actor_loss:.6f}\n"
            f"\tAvg Alpha Loss: {avg_alpha_loss:.6f}\n"
            f"\tAvg Alpha: {avg_alpha:.6f}\n"
            f"\tAvg Entropy: {avg_entropy:.6f}\n"
        )
        if avg_initial_q != "N/A":
            log_str += f"\tAvg Initial Q: {avg_initial_q:.6f}\n"

        log_str += (
            f"\tAvg Reward: {avg_reward:.6f}\n"
            f"--------------------------------------------------\n"
        )

        print(log_str)
        with open("log.txt", "a") as f:
            f.write(log_str)

    def _clear_losses(self):
        self.critic_losses.clear()
        self.actor_losses.clear()
        self.alpha_losses.clear()
        self.rewards.clear()
        self.alphas.clear()
        self.entropies.clear()
        self.initial_q_values.clear()

    def _train_critic(self, batch):
        raise NotImplementedError

    def _train_actor(self, batch):
        raise NotImplementedError

    def _update_target_networks(self):
        raise NotImplementedError

    def _get_alpha(self) -> torch.Tensor:
        if self.args.auto_alpha:
            return self.log_alpha.exp()
        else:
            return torch.tensor(self.args.alpha, device=self.args.device)

    def _update_alpha(self, probs: torch.Tensor):
        """alphaの更新"""
        if self.args.auto_alpha:
            if self.args.action_type == "discrete":
                # probs: (batch*T_valid, action_dim)
                # シャノンエントロピー H(p) = -Σ p * log(p)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            else:  # continuous
                # probs は実際には Normal distribution object
                entropy = probs.entropy()

            self.entropies.append(entropy.mean().item())
            # alphaの損失関数
            alpha_loss = (
                self.log_alpha * (entropy - self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            if self.args.use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    [self.log_alpha], self.args.alpha_max_grad_norm
                )
            self.alpha_optimizer.step()
            return self.log_alpha.exp().item(), alpha_loss.item()
        else:
            return self.args.alpha, 0.0

    def _create_action_env(self, batch):
        # 離散的なときはone-hotコード化
        if self.args.action_type == "discrete":
            actions = torch.nn.functional.one_hot(
                batch["actions"].squeeze(-1).long(),
                num_classes=self.policy[0].action_dim,
            ).float()
        else:
            actions = batch["actions"]

        # (batch, T, N, A) -> (batch, T, N*A)
        actions_env = actions.view(self.args.batch_size, self.args.episode_length, -1)

        assert actions_env.shape == (
            self.args.batch_size,
            self.args.episode_length,
            self.args.num_agents * self.policy[0].action_dim,
        )
        return actions_env
