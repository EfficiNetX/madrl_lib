import torch
import numpy as np
from algorithms.utils.rnn import RNNLayer


class QMIXPolicy:
    def __init__(self, args, obs_dim, action_dim, n_agents):
        self.args = args
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 各エージェントのQネットワーク（パラメータ共有 or 個別）
        self.agent_q_network = RNNLayer(
            inputs_dim=obs_dim,
            outputs_dim=action_dim,
            recurrent_N=args.recurrent_N,
            use_orthogonal=args.use_orthogonal,
        )
        # epsilon-greedy探索用
        self.epsilon = args.qmix_epsilon_start

        # hidden state管理
        self.hidden_states = None

    def init_hidden(self, batch_size):
        # 各エージェントの初期hidden stateを生成
        self.hidden_states = torch.zeros(
            batch_size, self.n_agents, self.agent_q_network.hidden_dim
        )

    def forward(self, obs, hidden_states):
        # obs: [batch, n_agents, obs_dim]
        # hidden_states: [batch, n_agents, hidden_dim]
        q_values, next_hidden_states = self.agent_q_network(obs, hidden_states)
        return q_values, next_hidden_states

    def select_actions(
        self, obs, hidden_states, avail_actions=None, test_mode=False
    ):
        # Q値計算
        q_values, next_hidden_states = self.forward(obs, hidden_states)
        # epsilon-greedyで行動選択
        actions = self._epsilon_greedy(q_values, avail_actions, test_mode)
        return actions, next_hidden_states

    def _epsilon_greedy(self, q_values, avail_actions=None, test_mode=False):
        # q_values: [batch, n_agents, action_dim]
        if test_mode or self.epsilon <= 0.0:
            actions = q_values.argmax(dim=-1)
        else:
            actions = []
            for agent_q in q_values:
                if np.random.rand() < self.epsilon:
                    if avail_actions is not None:
                        avail = avail_actions[agent_q]
                        action = np.random.choice(np.where(avail)[0])
                    else:
                        action = np.random.randint(self.action_dim)
                else:
                    action = agent_q.argmax().item()
                actions.append(action)
            actions = torch.tensor(actions)
        return actions

    def update_epsilon(self, t_env):
        # εの減衰
        eps_start = self.args.qmix_epsilon_start
        eps_end = self.args.qmix_epsilon_finish
        eps_anneal = self.args.qmix_epsilon_anneal_time
        self.epsilon = max(
            eps_end, eps_start - (eps_start - eps_end) * t_env / eps_anneal
        )

    def parameters(self):
        return self.agent_q_network.parameters()

    def save_models(self, path):
        torch.save(self.agent_q_network.state_dict(), f"{path}/agent_q.th")

    def load_models(self, path):
        self.agent_q_network.load_state_dict(torch.load(f"{path}/agent_q.th"))
