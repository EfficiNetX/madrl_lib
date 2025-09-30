import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.rnn import RNNLayer
from utils.util import update_linear_schedule


class QMIXPolicy:
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.args = args
        self.lr = args.lr
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.qmix_rnn_hidden_dim = args.qmix_rnn_hidden_dim

        # Qネットワーク
        self.agent_q_network = RNNAgent(
            self.obs_space, self.qmix_rnn_hidden_dim, self.action_space
        )
        self.q_out = nn.Linear(self.qmix_rnn_hidden_dim, self.action_space)
        self.epsilon = args.qmix_epsilon_start

        self.optimizer = torch.optim.Adam(
            list(self.agent_q_network.parameters())
            + list(self.q_out.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def forward_q_network(self, obs, actions=None, filled=None) -> np.ndarray:
        # obs: (batch_size, episode_length, n_agents, obs_dim)
        batch_size, episode_length, n_agents, obs_dim = obs.shape
        device = obs.device if hasattr(obs, "device") else "cpu"
        q_list = []
        hxs = torch.zeros(
            batch_size * n_agents, self.qmix_rnn_hidden_dim, device=device
        )
        for t in range(episode_length):
            obs_t = obs[:, t].reshape(batch_size * n_agents, obs_dim)
            # filledがあればマスクとして利用（0埋め部分はhidden stateリセットではなく学習で除外）
            if filled is not None:
                mask_t = (
                    filled[:, t].repeat(1, n_agents).reshape(-1, 1).to(device)
                )
            else:
                mask_t = torch.ones(batch_size * n_agents, 1, device=device)
            rnn_out, hxs = self.agent_q_network(obs_t, hxs, mask_t)
            q_t = self.q_out(rnn_out)  # (batch_size*n_agents, action_dim)
            q_t = q_t.view(batch_size, n_agents, self.action_space)
            q_list.append(q_t)
        q_values = torch.stack(
            q_list, dim=1
        )  # (batch_size, episode_length, n_agents, action_dim)
        if actions is not None:
            actions = actions.squeeze(-1)
            chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            return (
                chosen_q.cpu().detach().numpy()
            )  # np.arrayで返す # (batch_size, episode_length, n_agents)
        else:
            return (
                q_values.cpu().detach().numpy()
            )  # np.arrayで返す # (batch_size, episode_length, n_agents, action_dim)

    def get_actions(
        self,
        states,
        hidden_states=None,
        masks=None,
        epsilon=None,
        deterministic=False,
    ):
        batch_size, n_agents, obs_dim = states.shape
        device = states.device if hasattr(states, "device") else "cpu"
        if epsilon is None:
            epsilon = self.epsilon
        hxs = (
            hidden_states
            if hidden_states is not None
            else torch.zeros(
                batch_size * n_agents, self.qmix_rnn_hidden_dim, device=device
            )
        )
        mask = (
            masks
            if masks is not None
            else torch.ones(batch_size * n_agents, 1, device=device)
        )
        rnn_out, next_hxs = self.agent_q_network(
            states.reshape(batch_size * n_agents, obs_dim), hxs, mask
        )
        q_values = self.q_out(rnn_out).view(
            batch_size, n_agents, self.action_space
        )
        if deterministic:
            actions = q_values.argmax(dim=-1)
        else:
            if np.random.rand() < epsilon:
                actions = torch.randint(
                    0, self.action_space, (batch_size, n_agents), device=device
                )
            else:
                actions = q_values.argmax(dim=-1)
        return actions, next_hxs

    def lr_decay(self, episode, total_episodes):
        update_linear_schedule(
            self.optimizer, episode, total_episodes, self.lr
        )


class RNNAgent(nn.Module):
    def __init__(self, obs_space, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.fc1 = nn.Linear(obs_space, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)
        self.rnn_hidden_dim = rnn_hidden_dim

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.rnn_hidden_dim)

    def forward(self, inputs, hidden_state, mask=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        if mask is not None:
            h_in = h_in * mask
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
