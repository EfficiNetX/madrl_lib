import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.rnn import RNNLayer
from utils.util import update_linear_schedule
from typing import Tuple


class QPolicy:  # QMIXPoliciesにしたほうがわかりやすい可能性がある
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.hidden_size = args.hidden_size
        self.obs_dim = len(obs_space)  # 観測の次元数
        self.share_obs_dim = len(share_obs_space)  # shared観測の次元数
        self.action_dim = len(
            action_space
        )  # 行動の種類数(transfomer_policy.pyのaction_dimの定義に倣う)

        self.agent_q_network = RNNAgent(self.obs_dim, self.hidden_size, self.action_dim, self.args)
        self.agent_q_network.to(self.args.device)  # モデルをデバイスに移動
        self.epsilon = args.epsilon_start

        self.optimizer = torch.optim.Adam(
            list(self.agent_q_network.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def forward(self, obs, dones):
        """
        QPolicyはforwardでRNNAgentのforwardのみ呼び出す（責務分離）
        """
        return self.agent_q_network.forward(obs, dones)

    def get_actions(
        self,
        obs,
        dones=None,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RNNAgentの出力（q_values）を受けて方針決定のみ担当
        """
        q_values = self.agent_q_network.forward(obs, dones)
        batch_size, n_agents, action_dim = q_values.shape
        if deterministic:
            actions = torch.argmax(q_values, dim=-1)
        else:
            random_numbers = torch.rand(batch_size, n_agents, device=q_values.device)
            random_actions = torch.randint(
                0, action_dim, (batch_size, n_agents), device=q_values.device
            )
            greedy_actions = torch.argmax(q_values, dim=-1)
            actions = torch.where(random_numbers < self.epsilon, random_actions, greedy_actions)
        actions = actions.view(batch_size, n_agents, 1)
        return actions

    def init_hidden(self, batch_size):
        return self.agent_q_network.init_hidden(batch_size)

    def update_epsilon(self, t_env):
        self.epsilon = max(
            self.args.epsilon_final,
            self.args.epsilon_start * (1 - t_env / self.args.epsilon_anneal_time)
            + self.args.epsilon_final * (t_env / self.args.epsilon_anneal_time),
        )


class RNNAgent(nn.Module):  # エージェントのQネットワーク. 入力次元はobs_dim, 出力次元はaction_dim
    def __init__(self, obs_dim, rnn_hidden_dim, actions_dim, args):
        super(RNNAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, rnn_hidden_dim)  # 線形層
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # GRU層
        self.fc2 = nn.Linear(rnn_hidden_dim, actions_dim)  # 線形層
        self.rnn_hidden_dim = rnn_hidden_dim
        self.args = args
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = torch.zeros(
            batch_size, self.args.num_agents, self.rnn_hidden_dim, device=self.args.device
        )

    def forward(self, obs, dones):  # dones: 1ならばエピソード終わり
        # obs,hidden_statesをreshapeしてRNNに入力できる形に変換し、RNNの出力を元の形に戻す
        # 入力値の一部を表示
        # tensorの形状を変換
        batch_size, n_agents, obs_dim = obs.shape
        obs = obs.reshape(batch_size * n_agents, obs_dim)
        hxs = (
            self.hidden_states.reshape(batch_size * n_agents, self.args.hidden_size)
            if self.hidden_states is not None
            else torch.zeros(batch_size * n_agents, self.args.hidden_size, device=self.args.device)
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
        h = h.view(batch_size, n_agents, self.args.hidden_size)
        q = q.view(batch_size, n_agents, -1)  # (batch_size, n_agents, action_dim)次元に変換
        self.hidden_states = h.view(batch_size, n_agents, self.args.hidden_size)
        return q
