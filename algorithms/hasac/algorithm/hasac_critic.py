import torch

from algorithms.utils.mlp import MLPBase
from algorithms.utils.util import init


class Critic(MLPBase):
    def __init__(self, args, total_obs_dim, total_action_dim):
        super(Critic, self).__init__(
            args,
            total_obs_dim + total_action_dim,
            hidden_size=args.critic_hidden_size,
            layer=args.critic_layer_N,
        )
        self.args = args
        self.action_dim = total_action_dim

        # 初期化関数の設定
        if args.use_orthogonal:
            init_method = torch.nn.init.orthogonal_
        else:
            init_method = torch.nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: torch.nn.init.constant_(x, 0))

        self.q_out = init_(torch.nn.Linear(self.hidden_size, 1))

        self.to(args.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        隠れ層による特徴抽出の後に、行動分布のパラメータを出力する。
        """
        features = super(Critic, self).forward(obs)
        q_value = self.q_out(features)
        return q_value
