import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.rnn import RNNLayer
from utils.util import update_linear_schedule


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

        # Qネットワーク
        self.agent_q_network = RNNAgent(
            self.obs_dim, self.qmix_rnn_hidden_dim, self.action_dim
        )
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
        masks: (num_rollout_threads * n_agents, 1) # 1ならば有効
        学習をする場合
        obs: (qmix_batch_size, n_agents, obs_dim)
        hidden_states: (qmix_batch_size, n_agents, qmix_rnn_hidden_dim)
        masks: (qmix_batch_size * n_agents, 1) # 1ならば有効
        """
        batch_size, n_agents, obs_dim = obs.shape
        device = obs.device if hasattr(obs, "device") else "cpu"
        hxs = (
            hidden_states
            if hidden_states is not None
            else torch.zeros(
                batch_size * n_agents, self.qmix_rnn_hidden_dim, device=device
            )
        )
        mask = (
            dones
            if dones is not None
            else torch.zeros(batch_size * n_agents, 1, device=device)
        )
        q_values, next_hxs = self.agent_q_network(
            obs.reshape(batch_size * n_agents, obs_dim), hxs, mask
        )  # q_values: (batch_size * n_agents, action_space)
        q_values = q_values.view(batch_size, n_agents, self.action_space)
        next_hxs = next_hxs.view(
            batch_size, n_agents, self.qmix_rnn_hidden_dim
        )
        return q_values, next_hxs

    def get_actions(
        self,
        obs,
        hidden_states=None,
        dones=None,
        epsilon=None,
        deterministic=False,
    ):
        """
        データを集める時にしか使わないため，batch_sizeはnum_rollout_threadsと一致する
        obs: (batch_size, n_agents, obs_dim)
        hidden_states: (batch_size, n_agents, qmix_rnn_hidden_dim)
        masks: (batch_size * n_agents, 1) # 1ならば有効
        epsilon: (float) ε-greedyのε
        """
        if epsilon is None:
            epsilon = self.epsilon  # TODO: epsilon関係の実装を整理する
        q_values, next_hidden_states = self.forward(obs, hidden_states, dones)
        if deterministic:
            actions = torch.argmax(q_values, dim=-1)  # (batch_size, n_agents)
        else:
            batch_size, n_agents, action_dim = q_values.shape
            random_numbers = torch.rand(batch_size, n_agents).to(
                q_values.device
            )
            random_actions = torch.randint(
                0, action_dim, (batch_size, n_agents), device=q_values.device
            )
            greedy_actions = torch.argmax(
                q_values, dim=-1
            )  # (batch_size, n_agents)
            actions = torch.where(
                random_numbers < epsilon, random_actions, greedy_actions
            )  # (batch_size, n_agents)
        return actions, next_hidden_states

    def lr_decay(self, episode, total_episodes):
        update_linear_schedule(
            self.optimizer, episode, total_episodes, self.lr
        )

    def init_hidden(self, batch_size):

        return torch.zeros(
            batch_size, self.args.n_agents, self.qmix_rnn_hidden_dim
        )


class RNNAgent(nn.Module):
    def __init__(self, obs_space_dim, rnn_hidden_dim, actions_dim):
        super(RNNAgent, self).__init__()
        self.fc1 = nn.Linear(obs_space_dim, rnn_hidden_dim)  # 線形層
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # GRU層
        self.fc2 = nn.Linear(rnn_hidden_dim, actions_dim)  # 線形層
        self.rnn_hidden_dim = rnn_hidden_dim

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.rnn_hidden_dim)

    def forward(self, inputs, hidden_state, mask=None):  # mask: 1ならば有効
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        if mask is not None:
            h_in = h_in * mask
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
