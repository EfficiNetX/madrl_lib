import torch

from algorithms.hasac.algorithm.hasac_actor import Actor


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
        self.lr = args.actor_lr
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
        self.hidden_size = args.actor_hidden_size
        self.hidden_layer_N = args.actor_layer_N
        self.obs_dim = len(obs_space)
        self.share_obs_dim = len(share_obs_space)
        self.action_dim = len(action_space)
        self.action_shape = (
            1 if self.args.action_type == "discrete" else len(self.action_space)
        )
        # Actorクラスの生成
        self.actor = Actor(
            args=args,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        # Optimizerの生成
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.args.actor_lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )
        # デバイスの設定
        self.device = args.device
        self.actor.to(self.device)

    def get_action(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        行動を取得する。'(rollout時)'
        """
        action, action_env, _, _ = self._get_action_and_probs(
            obs, deterministic, type="rollout"
        )
        """
        assert action.shape == (
            self.args.num_rollout_threads,
            self.action_shape,
        )
        assert action_env.shape == (
            self.args.num_rollout_threads,
            self.action_dim,
        )
        """
        return action, action_env

    def get_action_with_probability(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        行動確率を取得する。
        """
        _, action_env, log_prob, probs = self._get_action_and_probs(
            obs, deterministic=False, type="train"
        )
        """
        assert action_env.shape == (
            self.args.batch_size,
            self.args.episode_length,
            self.action_dim,
        )
        assert log_prob.shape == (
            self.args.batch_size,
            self.args.episode_length,
        )
        assert probs.shape == (
            self.args.batch_size,
            self.args.episode_length,
            self.action_dim,
        )
        """
        return action_env, log_prob, probs

    def _get_action_and_probs(
        self, obs: torch.Tensor, deterministic=False, type="rollout"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if type != "rollout" and type != "train":
            raise ValueError(f"type should be 'rollout' or 'train', but got {type}")
        if self.args.action_type == "discrete":
            return self._get_action_and_probs_discrete(obs, deterministic, type)
        elif self.args.action_type == "continuous":
            return self._get_action_and_probs_continuous(obs, deterministic, type)
        else:
            raise (ValueError("action_type should be discrete or continuous."))

    def _get_action_and_probs_discrete(
        self, obs: torch.Tensor, deterministic: bool, type: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if deterministic and type == "rollout":  # 決定的
            action = torch.argmax(probs, dim=-1, keepdim=True)
            action_env = torch.nn.functional.one_hot(
                action.squeeze(-1), num_classes=self.action_dim
            ).float()
            log_prob = None  # type = "rollout"のときは不要
        elif not deterministic and type == "rollout":
            action = torch.nn.functional.gumbel_softmax(
                logits, tau=1.0, hard=True, dim=-1
            ).argmax(dim=-1, keepdim=True)
            action_env = torch.nn.functional.one_hot(
                action.squeeze(-1), num_classes=self.action_dim
            ).float()
            log_prob = None  # type = "rollout"のときは不要
        elif not deterministic:
            # gumbel-softmaxでサンプリング hard = Trueでone-hot化
            action_env = torch.nn.functional.gumbel_softmax(
                logits, tau=self.args.gumbel_softmax_tau, hard=False, dim=-1
            )
            action = torch.argmax(action_env, dim=-1, keepdim=True)
            log_prob = (
                (
                    torch.log(probs + 1e-10).clamp(
                        min=self.args.min_log_prob, max=self.args.max_log_prob
                    )
                )
                * action_env
            ).sum(dim=-1, keepdim=False)
        else:
            raise ValueError(
                "deterministic=True is only supported during rollout collection."
            )
        return action, action_env, log_prob, probs

    def _get_action_and_probs_continuous(
        self, obs: torch.Tensor, deterministic: bool = False, type: str = "rollout"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if deterministic and type == "rollout":
            z = mean
        elif not deterministic:
            z = normal.rsample()
        else:
            raise ValueError(
                "deterministic=True is only supported during rollout collection."
            )
        action = torch.tanh(z)
        action_env = action
        log_prob = (normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(
            dim=-1, keepdim=False
        )
        return action, action_env, log_prob, normal
