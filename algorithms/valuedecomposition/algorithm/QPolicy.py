import torch
import torch.nn as nn

from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.util import init


class QPolicy:
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
        self.obs_dim = len(obs_space)
        self.share_obs_dim = len(share_obs_space)
        self.action_dim = len(action_space)
        # 入力次元がobs_dim、出力次元がaction_dimのRNNを作成
        self.agent_q_network = RNNQNetwork(
            self.obs_dim, self.hidden_size, self.action_dim, self.args
        )
        self.agent_q_network.to(self.args.device)
        self.epsilon = args.epsilon_start

        self.optimizer = torch.optim.Adam(
            list(self.agent_q_network.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def get_actions(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        観測情報からQ値を計算して、方針決定を行う
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
            actions = torch.where(
                random_numbers < self.epsilon, random_actions, greedy_actions
            )
        actions = actions.view(batch_size, n_agents, 1)
        return actions

    def init_hidden(self, batch_size: int) -> None:
        return self.agent_q_network.init_hidden(batch_size)

    def update_epsilon(self, t_env: int) -> None:
        epsilon_anneal_time = self.args.num_env_steps * self.args.epsilon_anneal_ratio
        self.epsilon = max(
            self.args.epsilon_final,
            self.args.epsilon_start * (1 - t_env / epsilon_anneal_time)
            + self.args.epsilon_final * (t_env / epsilon_anneal_time),
        )


class RNNQNetwork(nn.Module):
    """
    MLPBase + RNNLayer を使用したQ-Network
    
    構成:
    - MLPBase: 特徴抽出（LayerNorm + 多層MLP）
    - RNNLayer: 時系列情報の処理（GRU + LayerNorm）
    - 出力層: Q値の計算
    """

    def __init__(self, obs_dim: int, hidden_size: int, action_dim: int, args):
        super(RNNQNetwork, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self._recurrent_N = args.recurrent_N

        # MLPBase: 入力の特徴抽出
        self.base = MLPBase(args, obs_dim)

        # RNNLayer: 時系列情報の処理
        self.rnn = RNNLayer(
            inputs_dim=hidden_size,      # MLPBaseからの出力次元
            outputs_dim=hidden_size,     # RNNの出力次元（Q出力層への入力）
            recurrent_N=self._recurrent_N,
            use_orthogonal=args.use_orthogonal,
        )

        # 出力層: Q値
        self._init_output_layer(hidden_size, action_dim, args)

        self.hidden_states = None

    def _init_output_layer(self, hidden_size: int, action_dim: int, args) -> None:
        """出力層の初期化"""
        if args.use_orthogonal:
            init_method = nn.init.orthogonal_
        else:
            init_method = nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=args.gain)

        self.q_out = init_(nn.Linear(hidden_size, action_dim))

    def init_hidden(self, batch_size: int) -> None:
        """隠れ状態の初期化"""
        self.hidden_states = torch.zeros(
            batch_size,
            self.args.num_agents,
            self._recurrent_N,
            self.hidden_size,
            device=self.args.device,
        )

    def forward(self, obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch_size, n_agents, obs_dim)
        dones: (batch_size, n_agents) or None
        Returns: (batch_size, n_agents, action_dim)
        """
        batch_size, n_agents, obs_dim = obs.shape

        # 隠れ状態が未初期化の場合は初期化
        if self.hidden_states is None:
            self.init_hidden(batch_size)

        # masksの準備（donesがNoneの場合は1で埋める）
        if dones is not None:
            masks = (~dones).float().reshape(batch_size * n_agents, 1)
        else:
            masks = torch.ones(batch_size * n_agents, 1, device=self.args.device)

        # (batch_size, n_agents, obs_dim) -> (batch_size * n_agents, obs_dim)
        obs_flat = obs.reshape(batch_size * n_agents, obs_dim)

        # 隠れ状態の取得
        hxs = self.hidden_states.reshape(
            batch_size * n_agents, self._recurrent_N, self.hidden_size
        )

        # MLPBase: 特徴抽出
        x = self.base(obs_flat)

        # RNNLayer: 時系列処理
        x, hxs = self.rnn(x, hxs, masks)

        # 出力層: Q値
        q_values = self.q_out(x)

        # 隠れ状態を保存
        self.hidden_states = hxs.reshape(
            batch_size, n_agents, self._recurrent_N, self.hidden_size
        )

        # 出力形状を元に戻す
        q_values = q_values.view(batch_size, n_agents, -1)

        return q_values
