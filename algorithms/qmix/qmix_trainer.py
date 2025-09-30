import copy
import torch
import torch.nn as nn
from torch.optim import RMSprop
import numpy as np


class QMIXTrainer:
    def __init__(self, policy, mixer, args, logger=None):
        self.args = args
        self.policy = policy  # QMIXPolicy
        self.mixer = mixer  # QMixer
        self.logger = logger

        self.params = list(policy.agent_q_network.parameters()) + list(
            mixer.parameters()
        )
        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)
        self.episode_num = 0  # 学習エピソード数

        self.last_target_update_episode = 0
        self.qmix_target_update_interval = args.qmix_target_update_interval
        self.log_stats_t = -self.args.learner_log_interval - 1

    # TODO: 学習率減衰関数を書く (関数自体はpolicyの中にある)

    def train(self, episode_batch):
        """
        episode_batch: dict of (np.ndarray)
            - 'share_obs': (batch_size, episode_length + 1, share_obs_dim)
            - 'obs': (batch_size, episode_length + 1, n_agents, obs_dim)
            - 'actions': (batch_size, episode_length, n_agents, 1)
            - 'rewards': (batch_size, episode_length, n_agents, 1)
            - 'dones': (batch_size, episode_length, 1)
            - 'filled': (batch_size, episode_length, 1) # パディングされていない部分を示すマスク 1ならば有効 0ならば無効
        """
        # numpy -> torch.Tensor
        share_obs = torch.tensor(
            episode_batch["share_obs"],
            dtype=torch.float32,
            device=self.args.device,
        )  # (batch_size, episode_length + 1, share_obs_dim)
        obs = torch.tensor(
            episode_batch["obs"], dtype=torch.float32, device=self.args.device
        )  # (batch_size, episode_length + 1, n_agents, obs_dim)
        actions = torch.tensor(
            episode_batch["actions"], dtype=torch.long, device=self.args.device
        )  # (batch_size, episode_length, n_agents, 1)
        rewards = torch.tensor(
            episode_batch["rewards"],
            dtype=torch.float32,
            device=self.args.device,
        )  # (batch_size, episode_length, n_agents, 1)
        dones = torch.tensor(
            episode_batch["dones"],
            dtype=torch.float32,
            device=self.args.device,
        )  # (batch_size, episode_length, 1)
        filled = torch.tensor(
            episode_batch["filled"],
            dtype=torch.float32,
            device=self.args.device,
        )  # (batch_size, episode_length, 1)
        self.episode_num += episode_batch["share_obs"].shape[0]
        # TD誤差を計算
        # ① 実際の行動のQ値を計算
        q_values = self.policy.forward_q_network(
            obs[:, :-1], actions
        )  # (batch_size, episode_length, n_agents)
        chosen_action_qvals = self.mixer.forward(
            q_values, share_obs[:, :-1]
        )  # (batch_size, episode_length, 1)
        # ② rewardsを計算
        rewards_sum = rewards.sum(dim=2)  # (batch_size, episode_length, 1)
        # ③ 次状態の各エージェントのmax Q値を計算
        with torch.no_grad():
            next_obs = obs[
                :, 1:
            ]  # (batch_size, episode_length, n_agents, obs_dim)
            all_next_q = self.target_policy.forward_q_network(
                next_obs
            )  # (batch_size, episode_length, n_agents, n_actions)
            max_next_q, _ = all_next_q.max(
                dim=-1
            )  # (batch_size, episode_length, n_agents)
            target_max_qvals = self.target_mixer.forward(
                max_next_q, share_obs[:, 1:]
            )  # (batch_size, episode_length, 1)

        # TD誤差を計算
        td_errors = chosen_action_qvals - (
            rewards_sum + self.args.qmix_gamma * target_max_qvals * (1 - dones)
        )
        # TD誤差の平均を取る
        loss = (td_errors**2 * filled).sum() / filled.sum()

        # 勾配を計算
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        # ターゲットネットワークの更新
        if (
            self.args.qmix_target_update_interval > 0
            and (self.episode_num - self.last_target_update_episode)
            >= self.qmix_target_update_interval
        ):
            self.last_target_update_episode = self.episode_num
            self._update_targets()

    def _update_targets(self):
        self.target_policy.agent_q_network.load_state_dict(
            self.policy.agent_q_network.state_dict()
        )
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")

    # TODO: 学習モードと推論モードの切り替え関数を書く
    """
    学習モードと推論モードの切り替え
    def prep_rollout(self):
        self.policy.eval()
        self.mixer.eval()

    def prep_training(self):
        self.policy.train()
        self.mixer.train()

    """
