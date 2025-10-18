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

        self.last_target_update_episode = 0
        self.qmix_target_update_interval = args.qmix_target_update_interval
        self.log_interval = args.log_interval
        self.learned_steps = 0
        self.loss_list = []
        self.reward_list = []

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
        self.learned_steps += 1
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
        avail_actions = torch.tensor(
            episode_batch["avail_actions"],
            dtype=torch.bool,
            device=self.args.device,
        )  # (batch_size, episode_length + 1, n_agents, action_dim)
        # TD誤差を計算
        # ① 実際の行動のQ値を計算
        total_q_values = []
        hidden_state = self.policy.init_hidden(
            self.args.qmix_batch_size
        )  # バッチ数でRNNの隠れ状態を初期化
        for t in range(self.args.episode_length):  # episode_length回ループ
            q_values, hidden_state = self.policy.forward(
                obs[:, t], hidden_state, None
            )
            total_q_values.append(q_values)
        total_q_values = torch.stack(
            total_q_values, dim=1
        )  # (batch_size, episode_length, n_agents, action_space)
        chosen_action_qvals = torch.gather(
            total_q_values, dim=3, index=actions
        ).squeeze(
            3
        )  # (batch_size, episode_length, n_agents)

        # ② ターゲットネットワークを用いて次の状態での最大Q値を計算
        target_total_q_values = []
        hidden_state = self.target_policy.init_hidden(
            self.args.qmix_batch_size
        )
        for t in range(self.args.episode_length):
            target_q_values, hidden_state = self.target_policy.forward(
                obs[:, t + 1], hidden_state, None
            )
            target_total_q_values.append(target_q_values)
            unavailable_actions = (
                avail_actions[:, t + 1] == 0
            )  # (batch, n_agents, action_dim)
            target_total_q_values[t][unavailable_actions] = -1e10
        # avail_actionsを考慮して，選択できない行動のQ値を大きな負の値にする
        target_total_q_values = torch.stack(
            target_total_q_values, dim=1
        )  # (batch_size, episode_length, n_agents, action_space)
        target_max_qvals = target_total_q_values.max(dim=3)[
            0
        ]  # (batch_size, episode_length, n_agents)

        # ③ ミキサーを用いて全体のQ値を計算
        if self.mixer is not None:
            mixed_chosen_action_qvals = self.mixer(
                chosen_action_qvals, share_obs[:, :-1]
            )
            mixed_target_max_qvals = self.target_mixer(
                target_max_qvals, share_obs[:, 1:]
            )  # (batch_size, episode_length, 1)
        else:  # VDNの場合
            mixed_chosen_action_qvals = chosen_action_qvals.sum(
                dim=2, keepdim=True
            )  # (batch_size, episode_length, 1)
            mixed_target_max_qvals = target_max_qvals.sum(
                dim=2, keepdim=True
            )  # (batch_size, episode_length, 1)

        # rewards (batch, episode_length, num_agents,1)を、(batch,episode_length,1)にする
        rewards = rewards.sum(dim=2)  # (batch_size, episode_length, 1)
        # 各バッチごとのTD誤差を計算
        # If any agent is not done, mask should be 1 (i.e., not all done)
        not_all_done = (
            (~dones).any(dim=2, keepdim=True).float()
        )  # (batch, episode_length, 1)
        target = (
            rewards
            + self.args.qmix_gamma * mixed_target_max_qvals * not_all_done
        )
        td_errors = mixed_chosen_action_qvals - target.detach()
        # 1ステップのTD誤差の２乗の平均値を取得する
        loss = (td_errors**2 * (~mask).unsqueeze(-1)).sum() / (~mask).sum()
        if self.learned_steps % self.log_interval == 0:
            print("TD Loss:", loss.item())
        # あとでpythonでvisualize化できるようにtxtにログを保存
        # 20回ごとのloss1の平均を保存
        self.loss_list.append(loss.item())
        self.reward_list.append(rewards.sum().item() / rewards.shape[0])

        if self.learned_steps % 50 == 0:
            avg_loss = sum(self.loss_list) / len(self.loss_list)
            avg_reward = sum(self.reward_list) / len(self.reward_list)
            with open("loss_log.txt", "a") as f:
                f.write(f"{self.learned_steps},{avg_loss},{avg_reward}\n")
            self.loss_list = []
            self.reward_list = []

        # 勾配を計算
        self.policy.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(
            self.policy.agent_q_network.parameters(), self.args.grad_norm_clip
        )
        self.policy.optimizer.step()

        # ターゲットネットワークの更新
        if (
            self.args.qmix_target_update_interval > 0
            and (self.learned_steps - self.last_target_update_episode)
            >= self.qmix_target_update_interval
        ):
            print("Updated target network")
            self.last_target_update_episode = self.learned_steps
            self._update_targets()

    def _update_targets(self):
        self.target_policy.agent_q_network.load_state_dict(
            self.policy.agent_q_network.state_dict()
        )
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")
