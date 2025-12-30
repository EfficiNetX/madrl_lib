import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.utils.mlp import MLPBase
from algorithms.utils.util import init


class QMixer(nn.Module):
    """
    QMIXのMixingネットワーク
    
    HyperNetworkにMLPBaseを使用し、頑健な初期化と正規化を実現
    """

    def __init__(self, args, obs_space, share_obs_space, action_space):
        super(QMixer, self).__init__()
        self.args = args
        self.num_agents = args.num_agents
        self.obs_dim = len(obs_space)
        self.shared_obs_dim = len(share_obs_space)
        self.embed_dim = args.mixer_embed_dim
        self.hypernet_layers = getattr(args, "hypernet_layers", 1)

        # HyperNetwork for weights
        self._build_hypernetworks()

        # Bias hyper network
        self.hyper_b_1 = self._build_bias_network()

        # V network for state value
        self.V = self._build_v_network()

        self.to(self.args.device)

        # オプティマイザ
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def _build_hypernetworks(self) -> None:
        """HyperNetworkの構築"""
        if self.hypernet_layers == 1:
            # 1層の場合: シンプルな線形層
            self.hyper_w_1 = self._init_linear(
                self.shared_obs_dim, self.embed_dim * self.num_agents
            )
            self.hyper_w_final = self._init_linear(self.shared_obs_dim, self.embed_dim)
        elif self.hypernet_layers == 2:
            # 2層の場合: MLPBaseを使用
            hypernet_embed = self.args.hypernet_embed_dim
            self.hyper_w_1 = nn.Sequential(
                MLPBase(
                    self.args,
                    self.shared_obs_dim,
                    hidden_size=hypernet_embed,
                    layer=0,  # MLPLayerの追加層は0
                ),
                self._init_linear(hypernet_embed, self.embed_dim * self.num_agents),
            )
            self.hyper_w_final = nn.Sequential(
                MLPBase(
                    self.args,
                    self.shared_obs_dim,
                    hidden_size=hypernet_embed,
                    layer=0,
                ),
                self._init_linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise ValueError(
                f"hypernet_layers must be 1 or 2, got {self.hypernet_layers}"
            )

    def _build_bias_network(self) -> nn.Module:
        """バイアスネットワークの構築"""
        return self._init_linear(self.shared_obs_dim, self.embed_dim)

    def _build_v_network(self) -> nn.Module:
        """状態価値ネットワークの構築"""
        return nn.Sequential(
            MLPBase(
                self.args,
                self.shared_obs_dim,
                hidden_size=self.embed_dim,
                layer=0,
            ),
            self._init_linear(self.embed_dim, 1),
        )

    def _init_linear(self, in_features: int, out_features: int) -> nn.Linear:
        """初期化された線形層を生成"""
        if self.args.use_orthogonal:
            init_method = nn.init.orthogonal_
        else:
            init_method = nn.init.xavier_uniform_

        def init_(m):
            return init(
                m, init_method, lambda x: nn.init.constant_(x, 0), gain=self.args.gain
            )

        return init_(nn.Linear(in_features, out_features))

    def forward(
        self, agent_qs: torch.Tensor, shared_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        agent_qs: (batch_size, episode_length, n_agents) 各エージェントのQ値
        shared_obs: (batch_size, episode_length, shared_obs_dim) 共有観測
        return: (batch_size, episode_length, 1) 全体のQ値
        """
        batch_size = agent_qs.size(0)
        seq_len = agent_qs.size(1) if agent_qs.dim() > 2 else 1

        # Flatten for processing
        shared_obs = shared_obs.reshape(-1, self.shared_obs_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)

        # First layer weights and biases
        w1 = torch.abs(self.hyper_w_1(shared_obs))
        b1 = self.hyper_b_1(shared_obs)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        # Hidden layer
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Final layer weights
        w_final = torch.abs(self.hyper_w_final(shared_obs))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State value
        v = self.V(shared_obs).view(-1, 1, 1)

        # Output
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(batch_size, seq_len, 1)

        return q_tot
