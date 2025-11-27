import torch

from algorithms.utils.mlp import MLPBase
from algorithms.utils.util import init


class Actor(MLPBase):
    def __init__(self, args, obs_dim, action_dim):
        super(Actor, self).__init__(
            args, obs_dim, hidden_size=args.actor_hidden_size, layer=args.actor_layer_N
        )
        self.args = args
        self.action_dim = action_dim

        # 初期化関数の設定
        if args.use_orthogonal:
            init_method = torch.nn.init.orthogonal_
        else:
            init_method = torch.nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: torch.nn.init.constant_(x, 0))

        if self.args.action_type == "continuous":
            # 初期化を適用
            self.mean_linear = init_(torch.nn.Linear(self.hidden_size, self.action_dim))
            self.log_std_linear = init_(
                torch.nn.Linear(self.hidden_size, self.action_dim)
            )
        elif self.args.action_type == "discrete":
            # 離散行動空間の場合の出力層
            self.action_out = init_(torch.nn.Linear(self.hidden_size, self.action_dim))
        else:
            raise ValueError("action_type must be 'continuous' or 'discrete'.")

        self.to(args.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        隠れ層による特徴抽出の後に、行動分布のパラメータを出力する。
        """
        features = super(Actor, self).forward(obs)
        if self.args.action_type == "discrete":
            # 離散行動空間の場合
            logits = self.action_out(features)  # (..., action_dim)
            return logits
        elif self.args.action_type == "continuous":
            mean = self.mean_linear(features)
            log_std = self.log_std_linear(features)
            log_std = torch.clamp(
                log_std,
                min=self.args.min_log_std,
                max=self.args.max_log_std,
            )
            return mean, log_std
        else:
            raise ValueError("action_type must be 'continuous' or 'discrete'.")
