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
        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)
        self.episode_num = 0  # 学習エピソード数

        self.last_target_update_episode = 0
        self.qmix_target_update_interval = args.qmix_target_update_interval

    # TODO: 学習率減衰関数を書く

    def train(self, episode_batch):
        """
        episode_batch: dict of (np.ndarray)
            - 'share_obs': (batch_size, episode_length + 1, share_obs_dim)
            - 'obs': (batch_size, episode_length + 1, n_agents, obs_dim)
            - 'actions': (batch_size, episode_length, n_agents, 1)
            - 'rewards': (batch_size, episode_length, n_agents)
            - 'dones': (batch_size, episode_length, n_agents)
            - 'mask': (batch_size, episode_length) # パディングされていない部分を示すマスク 1ならば有効 0ならば無効
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
            dtype=torch.bool,
            device=self.args.device,
        )  # (batch_size, episode_length, num_agents)
        mask = torch.tensor(
            episode_batch["mask"],
            dtype=torch.bool,
            device=self.args.device,
        )  # (batch_size, episode_length)
        self.episode_num += episode_batch["share_obs"].shape[0]
        # TD誤差を計算
        # ① 実際の行動のQ値を計算
        total_q_values = []
        hidden_state = self.policy.init_hidden(
            self.args.qmix_batch_size
        )  # バッチ数でRNNの隠れ状態を初期化
        for t in range(self.args.episode_length):  # episode_length回ループ
            q_values, hidden_state = self.policy.forward(obs[:, t], hidden_state, None)
            total_q_values.append(q_values)
        total_q_values = torch.stack(
            total_q_values, dim=1
        )  # (batch_size, episode_length, n_agents, action_space)
        chosen_action_qvals = torch.gather(total_q_values, dim=3, index=actions).squeeze(
            3
        )  # (batch_size, episode_length, n_agents)

        # ② ターゲットネットワークを用いて次の状態での最大Q値を計算
        target_total_q_values = []
        hidden_state = self.target_policy.init_hidden(self.args.qmix_batch_size)
        for t in range(self.args.episode_length):
            target_q_values, hidden_state = self.target_policy.forward(
                obs[:, t + 1], hidden_state, None
            )
            target_total_q_values.append(target_q_values)
        target_total_q_values = torch.stack(
            target_total_q_values, dim=1
        )  # (batch_size, episode_length, n_agents, action_space)
        target_max_qvals = target_total_q_values.max(dim=3)[
            0
        ]  # (batch_size, episode_length, n_agents)

        # ③ ミキサーを用いて全体のQ値を計算
        if self.mixer is not None:
            mixed_chosen_action_qvals = self.mixer(chosen_action_qvals, share_obs[:, :-1])
            mixed_target_max_qvals = self.target_mixer(
                target_max_qvals, share_obs[:, 1:]
            )  # (batch_size, episode_length, 1)
        else:  # VDNの場合
            mixed_chosen_action_qvals = chosen_action_qvals.sum(dim=2)
            mixed_target_max_qvals = target_max_qvals.sum(dim=2)

        # rewards (batch, episode_length, num_agents,1)を、(batch,episode_length,1)にする
        rewards = rewards.sum(dim=2)  # (batch_size, episode_length, 1)
        # 各バッチごとのTD誤差を計算
        target = rewards + self.args.qmix_gamma * mixed_target_max_qvals * mask.unsqueeze(-1)
        td_errors = mixed_chosen_action_qvals - target.detach()
        # 1ステップのTD誤差の２乗の平均値を取得する
        loss = (td_errors**2 * (~mask).unsqueeze(-1)).sum() / (~mask).sum()

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
        self.target_policy.agent_q_network.load_state_dict(self.policy.agent_q_network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")
