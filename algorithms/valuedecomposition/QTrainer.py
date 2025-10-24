import copy
import torch


class QTrainer:
    def __init__(self, policy, mixer, args, logger=None):
        self.args = args
        self.policy = policy
        self.mixer = mixer
        self.logger = logger
        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)

        self.last_target_update_episode = 0
        self.target_network_update_interval = args.target_network_update_interval
        self.log_interval = args.log_interval
        self.learned_steps = 0
        self.loss_list = []
        self.reward_list = []

    def train(self, episode_batch: dict) -> None:
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
        obs_t0 = batch["obs"][:, :-1]
        mask_t0 = batch["mask"][:, :-1]
        total_q_values = self._collect_qval(self.policy, obs_t0, mask_t0)
        chosen_action_qvals = torch.gather(total_q_values, dim=3, index=batch["actions"]).squeeze(3)

        obs_t1 = batch["obs"][:, 1:]
        mask_t1 = batch["mask"][:, 1:]
        target_total_q_values = self._collect_qval(self.target_policy, obs_t1, mask_t1)
        for t in range(self.args.episode_length):
            unavailable_actions = batch["avail_actions"][:, t + 1] == 0
            target_total_q_values[:, t][unavailable_actions] = -1e10
        target_max_qvals = target_total_q_values.max(dim=3)[0]

        mixed_chosen_action_qvals = self._mix_q_values(
            chosen_action_qvals, batch["share_obs"][:, :-1], self.mixer
        )
        mixed_target_max_qvals = self._mix_q_values(
            target_max_qvals, batch["share_obs"][:, 1:], self.target_mixer
        )

        rewards = batch["rewards"].sum(dim=2)
        not_all_done = (~batch["dones"]).any(dim=2, keepdim=True).float()
        target = rewards + self.args.gamma * mixed_target_max_qvals * not_all_done
        td_errors = mixed_chosen_action_qvals - target.detach()
        loss = (td_errors**2 * (~batch["mask"]).unsqueeze(-1)).sum() / (~batch["mask"]).sum()
        self._log_training_progress(loss.item(), rewards.sum().item() / rewards.shape[0])

        self.policy.optimizer.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.policy.optimizer.step()
        if self._should_update_targets():
            self._update_targets()

    def _collect_qval(
        self, network: torch.nn.Module, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        obs: (batch_size, episode_length, n_agents, obs_dim)
        mask: (batch_size, episode_length) True=invalid, False=valid
        Returns: (batch_size, episode_length, n_agents, action_space)
        """
        total_q_values = []
        network.init_hidden(obs.shape[0])
        for t in range(obs.shape[1]):
            # mask=Trueの時点（エピソード終了後）でhidden stateをリセット
            done_mask = mask[:, t].unsqueeze(-1).float()
            q_values = network.forward(obs[:, t], done_mask)
            total_q_values.append(q_values)
        total_q_values = torch.stack(total_q_values, dim=1)
        return total_q_values

    def _should_update_targets(self) -> bool:
        """ターゲットネットワークの更新が必要か判定"""
        return (
            self.args.target_network_update_interval > 0
            and (self.learned_steps - self.last_target_update_episode)
            >= self.target_network_update_interval
        )

    def _update_targets(self) -> None:
        """ターゲットネットワークの更新を実行"""
        print("Updated target network")
        self.last_target_update_episode = self.learned_steps
        self.__update_targets()

    def __update_targets(self) -> None:
        self.target_policy.agent_q_network.load_state_dict(self.policy.agent_q_network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")

    def _clip_gradients(self) -> None:
        """QネットワークとMixerの勾配クリッピング処理"""
        torch.nn.utils.clip_grad_norm_(
            self.policy.agent_q_network.parameters(), self.args.max_grad_norm
        )
        if self.mixer is not None:
            torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.args.max_grad_norm)

    def _mix_q_values(
        self,
        agent_qvals: torch.Tensor,
        share_obs: torch.Tensor,
        mixer: torch.nn.Module,
    ) -> torch.Tensor:
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

    def _log_training_progress(self, loss: float, avg_reward: float) -> None:
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
