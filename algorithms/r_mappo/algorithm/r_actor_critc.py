import torch
import torch.nn as nn

from algorithms.utils.act import ACTLayer
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.util import check, init


class R_Actor(nn.Module):
    def __init__(
        self,
        args,
        obs_space,
        action_space,
    ):
        super(R_Actor, self).__init__()

        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=args.device)

        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_dim = len(self.obs_space)

        base = MLPBase
        self.base = base(
            args=args,
            obs_dim=self.obs_dim,
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                inputs_dim=self.hidden_size,
                outputs_dim=self.hidden_size,
                recurrent_N=self._recurrent_N,
                use_orthogonal=self._use_orthogonal,
            )
        self.act = ACTLayer(
            action_space=action_space,
            inputs_dim=self.hidden_size,
            use_orthogonal=self._use_orthogonal,
            gain=self._gain,
            args=args,
        )

        self.to(args.device)
        self.algo = args.algorithm_name

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(
                x=actor_features,
                hxs=rnn_states,
                masks=masks,
            )

        actions, action_log_probs = self.act(
            actor_features,
            available_actions,
            deterministic,
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Compute log probability and entropy of given actions."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            x=actor_features,
            action=action,
            available_actions=available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=args.device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        base = MLPBase

        self.cent_obs_dim = len(share_obs_space)

        self.base = base(
            args=args,
            obs_dim=self.cent_obs_dim,
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                inputs_dim=self.hidden_size,
                outputs_dim=self.hidden_size,
                recurrent_N=self._recurrent_N,
                use_orthogonal=self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(args.device)

    def forward(
        self,
        shared_obs,
        rnn_states,
        masks,
    ):
        shared_obs = check(shared_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(shared_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(
                x=critic_features,
                hxs=rnn_states,
                masks=masks,
            )

        values = self.v_out(critic_features)

        return values, rnn_states
