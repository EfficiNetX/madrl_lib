import copy
import torch


class QTrainer:
    def __init__(self, policy, mixer, args, logger=None):
        self.args = args
        self.policy = policy  # QPolicy
        self.mixer = mixer  # QMixer
        self.logger = logger
        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)

        self.last_target_update_episode = 0
        self.target_network_update_interval = args.target_network_update_interval
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
        batch = {
            "share_obs": torch.tensor(episode_batch["share_obs"], device=self.args.device),
            "obs": torch.tensor(episode_batch["obs"], device=self.args.device),
            "actions": torch.tensor(episode_batch["actions"], device=self.args.device),
            "rewards": torch.tensor(episode_batch["rewards"], device=self.args.device),
            "dones": torch.tensor(episode_batch["dones"], device=self.args.device),
            "mask": torch.tensor(episode_batch["mask"], device=self.args.device),
            "avail_actions": torch.tensor(episode_batch["avail_actions"], device=self.args.device),
        }

        # TD誤差を計算
        # ① 実際の行動のQ値を計算
        total_q_values = self._collect_qval(self.policy, batch["obs"][:, :-1])
        chosen_action_qvals = torch.gather(total_q_values, dim=3, index=batch["actions"]).squeeze(3)

        # ② ターゲットネットワークを用いて次の状態での最大Q値を計算
        target_total_q_values = self._collect_qval(self.target_policy, batch["obs"][:, 1:])
        # avail_actionsを考慮して，選択できない行動のQ値を
        # 大きな負の値にする
        for t in range(self.args.episode_length):
            # (batch, n_agents, action_dim)
            unavailable_actions = batch["avail_actions"][:, t + 1] == 0
            target_total_q_values[:, t][unavailable_actions] = -1e10
        target_max_qvals = target_total_q_values.max(dim=3)[
            0
        ]  # (batch_size, episode_length, n_agents)

        # ③ ミキサーを用いて全体のQ値を計算
        mixed_chosen_action_qvals = self._mix_q_values(
            chosen_action_qvals, batch["share_obs"][:, :-1], self.mixer
        )
        mixed_target_max_qvals = self._mix_q_values(
            target_max_qvals, batch["share_obs"][:, 1:], self.target_mixer
        )

        rewards = batch["rewards"].sum(dim=2)  # (batch_size, episode_length, 1)
        # 各バッチごとのTD誤差を計算
        # If any agent is not done, mask should be 1 (i.e., not all done)
        not_all_done = (
            (~batch["dones"]).any(dim=2, keepdim=True).float()
        )  # (batch, episode_length, 1)
        target = rewards + self.args.gamma * mixed_target_max_qvals * not_all_done
        td_errors = mixed_chosen_action_qvals - target.detach()
        # 1ステップのTD誤差の２乗の平均値を取得する
        loss = (td_errors**2 * (~batch["mask"]).unsqueeze(-1)).sum() / (~batch["mask"]).sum()
        self._log_training_progress(loss.item(), rewards.sum().item() / rewards.shape[0])

        # 勾配を計算
        self.policy.optimizer.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.policy.optimizer.step()

        # ターゲットネットワークの更新
        if self._should_update_targets():
            self._update_targets()

    def _collect_qval(self, network, obs):
        """
        obs: (batch_size, episode_length, n_agents, obs_dim) or (batch_size, episode_length+1, ...)
        Returns: (batch_size, episode_length, n_agents, action_space)
        """
        total_q_values = []
        network.init_hidden(obs.shape[0])
        for t in range(obs.shape[1]):
            q_values = network.forward(obs[:, t], None)
            total_q_values.append(q_values)
        total_q_values = torch.stack(total_q_values, dim=1)
        return total_q_values

    def _should_update_targets(self):
        """ターゲットネットワークの更新が必要か判定"""
        return (
            self.args.target_network_update_interval > 0
            and (self.learned_steps - self.last_target_update_episode)
            >= self.target_network_update_interval
        )

    def _update_targets(self):
        """ターゲットネットワークの更新を実行"""
        print("Updated target network")
        self.last_target_update_episode = self.learned_steps
        self.__update_targets()

    def __update_targets(self):
        self.target_policy.agent_q_network.load_state_dict(self.policy.agent_q_network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")

    def _clip_gradients(self):
        """QネットワークとMixerの勾配クリッピング処理"""
        torch.nn.utils.clip_grad_norm_(
            self.policy.agent_q_network.parameters(), self.args.max_grad_norm
        )
        if self.mixer is not None:
            torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.args.max_grad_norm)

    def _mix_q_values(self, agent_qvals, share_obs, mixer):
        """
        agent_qvals:
            (batch_size, episode_length, n_agents)
            or (batch_size, episode_length, n_agents, ...)
        share_obs: (batch_size, episode_length, share_obs_dim)
        mixer: mixer module or None
        Returns: (batch_size, episode_length, 1)
        """
        if mixer is not None:
            return mixer(agent_qvals, share_obs)
        else:
            return agent_qvals.sum(dim=2, keepdim=True)

    def _log_training_progress(self, loss, avg_reward):
        """
        ロスと報酬のログ出力・ファイル保存を分離
        """
        self.loss_list.append(loss)
        self.reward_list.append(avg_reward)
        if self.learned_steps % self.log_interval == 0:
            print(f"TD Loss: {loss}")
        if self.learned_steps % 50 == 0:
            avg_loss = sum(self.loss_list) / len(self.loss_list)
            avg_reward = sum(self.reward_list) / len(self.reward_list)
            with open("loss_log.txt", "a") as f:
                f.write(f"{self.learned_steps},{avg_loss},{avg_reward}\n")
            self.loss_list = []
            self.reward_list = []
