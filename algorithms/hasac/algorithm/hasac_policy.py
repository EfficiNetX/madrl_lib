import torch

from algorithms.utils.mlp import MLPBase
from algorithms.utils.util import init


class HASACPolicy:
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        # 引数の保存
        self.args = args
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.hidden_size = args.hidden_size
        self.hidden_layer_N = args.hidden_layer_N
        self.obs_dim = len(obs_space)
        self.share_obs_dim = len(share_obs_space)
        self.action_dim = len(action_space)
        # Actorクラスの生成
        self.actor = Actor(
            args=args,
            obs_dim=self.obs_dim,
        )
        # Optimizerの生成
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=args.opti_eps
        )
        # デバイスの設定
        self.device = args.device
        self.actor.to(self.device)

    @torch.no_grad()
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        行動をサンプリングまたは決定的に選択する。
        """
        if self.args.action_type == "discrete":
            logits = self.actor(obs)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
        elif self.args.action_type == "continuous":
            if deterministic:
                mean, _ = self.actor(obs)
                actions = mean
            else:
                mean, log_std = self.actor(obs)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                actions = dist.rsample()
                actions = torch.tanh(actions)  # 行動を[-1, 1]に制限
        else:
            raise ValueError("action_type must be 'continuous' or 'discrete'.")
        return actions


class Actor(MLPBase):
    def __init__(self, args, obs_dim, action_dim):
        super(Actor, self).__init__(args, obs_dim)
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
                min=self.args.hasac_log_std_min,
                max=self.args.hasac_log_std_max,
            )
            return mean, log_std
        else:
            raise ValueError("action_type must be 'continuous' or 'discrete'.")
