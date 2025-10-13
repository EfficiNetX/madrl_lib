import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.rnn import RNNLayer
from utils.util import update_linear_schedule
from typing import Tuple


class QMIXPolicy:  # QMIXPoliciesにしたほうがわかりやすい可能性がある
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.args = args
        self.qmix_batch_size = args.qmix_batch_size
        self.lr = args.lr
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.qmix_rnn_hidden_dim = args.qmix_rnn_hidden_dim
        self.obs_dim = len(obs_space)  # 観測の次元数
        self.share_obs_dim = len(share_obs_space)  # shared観測の次元数
        self.action_dim = len(
            action_space
        )  # 行動の種類数(transfomer_policy.pyのaction_dimの定義に倣う)

        self.agent_q_network = RNNAgent(
            self.obs_dim, self.qmix_rnn_hidden_dim, self.action_dim, self.args
        )
        self.agent_q_network.to(self.args.device)  # モデルをデバイスに移動
        self.epsilon = args.qmix_epsilon_start

        self.optimizer = torch.optim.Adam(
            list(self.agent_q_network.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def forward(self, obs, hidden_states, dones):
        """
        データを集める場合
        obs: (num_rollout_threads, n_agents, obs_dim)
        hidden_states: (num_rollout_threads, n_agents, qmix_rnn_hidden_dim)
        dones: (num_rollout_threads, n_agents, 1)
        学習をする場合
        obs: (qmix_batch_size, n_agents, obs_dim)
        hidden_states: (qmix_batch_size, n_agents, qmix_rnn_hidden_dim)
        dones: (qmix_batch_size, n_agents, 1)
        """
        batch_size, n_agents, obs_dim = obs.shape
        hxs = (
            hidden_states
            if hidden_states is not None
            else torch.zeros(
                batch_size, n_agents, self.qmix_rnn_hidden_dim, device=self.args.device
            )
        )
        dones = (
            dones
            if dones is not None
            else torch.zeros(batch_size, n_agents, 1, device=self.args.device)
        )
        q_values, next_hxs = self.agent_q_network(
            obs, hxs, dones
        )  # q_values: (batch_size * n_agents, action_dim)
        q_values = q_values.view(batch_size, n_agents, self.action_dim)
        next_hxs = next_hxs.view(batch_size, n_agents, self.qmix_rnn_hidden_dim)
        return q_values, next_hxs  # type: Tuple[torch.Tensor, torch.Tensor]

    def get_actions(
        self,
        obs,
        hidden_states=None,
        dones=None,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データを集める時にしか使わないため，batch_sizeはnum_rollout_threadsと一致する
        obs: (batch_size, n_agents, obs_dim)
        hidden_states: (batch_size, n_agents, qmix_rnn_hidden_dim)
        dones: (batch_size * n_agents, 1)
        epsilon: (float) ε-greedyのε
        """
        assert type(obs) is torch.Tensor, "obs should be a torch.Tensor"
        assert type(hidden_states) is torch.Tensor, "hidden_states should be a torch.Tensor"
        q_values, next_hidden_states = self.forward(obs, hidden_states, dones)
        if deterministic:
            actions = torch.argmax(q_values, dim=-1)  # (batch_size, n_agents)
        else:
            batch_size, n_agents, action_dim = q_values.shape
            random_numbers = torch.rand(batch_size, n_agents).to(q_values.device)
            random_actions = torch.randint(
                0, action_dim, (batch_size, n_agents), device=q_values.device
            )
            greedy_actions = torch.argmax(q_values, dim=-1)  # (batch_size, n_agents)
            actions = torch.where(
                random_numbers < self.epsilon, random_actions, greedy_actions
            )  # (batch_size, n_agents)
            actions = actions.view(batch_size, n_agents, 1)  # (batch_size, n_agents, 1)
        return actions, next_hidden_states

    def init_hidden(self, batch_size):

        return torch.zeros(
            batch_size, self.args.num_agents, self.qmix_rnn_hidden_dim, device=self.args.device
        )

    def update_epsilon(self, t_env):
        self.epsilon = max(
            self.args.qmix_epsilon_final,
            self.args.qmix_epsilon_start * (1 - t_env / self.args.qmix_epsilon_anneal_time)
            + self.args.qmix_epsilon_final * (t_env / self.args.qmix_epsilon_anneal_time),
        )


class RNNAgent(nn.Module):  # エージェントのQネットワーク. 入力次元はobs_dim, 出力次元はaction_dim
    def __init__(self, obs_dim, rnn_hidden_dim, actions_dim, args):
        super(RNNAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, rnn_hidden_dim)  # 線形層
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # GRU層
        self.fc2 = nn.Linear(rnn_hidden_dim, actions_dim)  # 線形層
        self.rnn_hidden_dim = rnn_hidden_dim
        self.args = args

    def forward(self, obs, hidden_states, dones):  # dones: 1ならばエピソード終わり
        # obs,hidden_statesをreshapeしてRNNに入力できる形に変換し、RNNの出力を元の形に戻す
        # 入力値の一部を表示
        # tensorの形状を変換
        batch_size, n_agents, obs_dim = obs.shape
        obs = obs.reshape(batch_size * n_agents, obs_dim)
        hxs = (
            hidden_states.reshape(batch_size * n_agents, self.args.qmix_rnn_hidden_dim)
            if hidden_states is not None
            else torch.zeros(
                batch_size * n_agents, self.args.qmix_rnn_hidden_dim, device=self.args.device
            )
        )
        dones = (
            dones.reshape(batch_size * n_agents, 1)
            if dones is not None
            else torch.zeros(batch_size * n_agents, 1, device=self.args.device)
        )
        x = F.relu(self.fc1(obs))
        hxs = hxs * (1 - dones.float())
        h = self.rnn(x, hxs)
        q = self.fc2(h)
        h = h.view(batch_size, n_agents, self.args.qmix_rnn_hidden_dim)
        q = q.view(batch_size, n_agents, -1)  # (batch_size, n_agents, action_dim)次元に変換
        return q, h
