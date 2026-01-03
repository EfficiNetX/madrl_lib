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
        self.target_policy.agent_q_network.rnn.rnn.flatten_parameters()

        self.last_target_update_episode = 0
        self.target_network_update_interval = args.target_network_update_interval
        self.log_interval = args.log_interval
        self.learned_steps = 0

    def train(self, episode_batch: dict) -> None:
        """
        episode_batch: dict of (np.ndarray)
            - 'share_obs': (batch_size, episode_length + 1, share_obs_dim)
            - 'obs': (batch_size, episode_length + 1, n_agents, obs_dim)
            - 'actions': (batch_size, episode_length, n_agents, 1)
            - 'rewards': (batch_size, episode_length, n_agents, 1)
            - 'dones': (batch_size, episode_length, n_agents)
            - 'avail_actions': (batch_size, episode_length + 1, n_agents, action_dim)
            - 'mask': (batch_size, episode_length) # 1ならばそのデータは無効（パディング）
        """
        self.learned_steps += 1
        batch = {
            "share_obs": torch.tensor(
                episode_batch["share_obs"], device=self.args.device
            ),
            "obs": torch.tensor(episode_batch["obs"], device=self.args.device),
            "actions": torch.tensor(episode_batch["actions"], device=self.args.device),
            "rewards": torch.tensor(episode_batch["rewards"], device=self.args.device),
            "dones": torch.tensor(episode_batch["dones"], device=self.args.device),
            "mask": torch.tensor(episode_batch["mask"], device=self.args.device),
            "avail_actions": torch.tensor(
                episode_batch["avail_actions"], device=self.args.device
            ),
        }
        obs_t0 = batch["obs"][:, :-1]
        total_q_values = self._collect_qval(self.policy, obs_t0)
        chosen_action_qvals = torch.gather(
            total_q_values, dim=3, index=batch["actions"]
        ).squeeze(3)

        obs_t1 = batch["obs"][:, 1:]
        target_total_q_values = self._collect_qval(self.target_policy, obs_t1)
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

        # rewards: (batch, ep_len, agents, 1) -> sum -> (batch, ep_len, 1)
        rewards = batch["rewards"].sum(dim=2)
        not_all_done = (~batch["dones"]).any(dim=2, keepdim=True).float()  # (batch, ep_len, 1)
        
        target = rewards + self.args.gamma * mixed_target_max_qvals * not_all_done
        td_errors = mixed_chosen_action_qvals - target.detach()
        loss = (td_errors**2 * (~batch["mask"]).unsqueeze(-1)).sum() / (
            ~batch["mask"]
        ).sum()

        # 勾配をリセット
        self.policy.optimizer.zero_grad()
        if self.mixer is not None:
            self.mixer.optimizer.zero_grad()

        loss.backward()

        if self.args.use_max_grad_norm:
            self._clip_gradients()

        # 各オプティマイザでパラメータ更新
        self.policy.optimizer.step()
        if self.mixer is not None:
            self.mixer.optimizer.step()
        if self._should_update_targets():
            self._update_targets()

    def _collect_qval(
        self, network: torch.nn.Module, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        各エージェントの各行動ごとのQ値を計算する
        network: Target Network or Policy Networkのどちらか
        obs: (batch_size, episode_length, n_agents, obs_dim)
        Returns: (batch_size, episode_length, n_agents, action_space)
        """
        total_q_values = []
        network.agent_q_network.init_hidden(obs.shape[0])
        for t in range(obs.shape[1]):
            q_values = network.agent_q_network.forward(obs[:, t], None)
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
        self.last_target_update_episode = self.learned_steps
        self.target_policy.agent_q_network.load_state_dict(
            self.policy.agent_q_network.state_dict()
        )
        # load_state_dict後にRNNの重みをフラット化
        self.target_policy.agent_q_network.rnn.rnn.flatten_parameters()
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        print("Updated target network")

    def _clip_gradients(self) -> None:
        """QネットワークとMixerの勾配クリッピング処理"""
        torch.nn.utils.clip_grad_norm_(
            self.policy.agent_q_network.parameters(), self.args.max_grad_norm
        )
        if self.mixer is not None:
            torch.nn.utils.clip_grad_norm_(
                self.mixer.parameters(), self.args.max_grad_norm
            )

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
        VDN(mixer=None)の場合は、各エージェントのQ値を合計する
        QMIX(mixer=QMix)の場合は、各エージェントのQ値を合計して、それをMixerに入力する
        """
        if mixer is not None:
            return mixer(agent_qvals, share_obs)
        else:
            return agent_qvals.sum(dim=2, keepdim=True)
