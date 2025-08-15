import torch
import torch.nn as nn

from algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from algorithms.utils.util import check
from utils.util import update_linear_schedule


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        super(TransformerPolicy, self).__init__()
        self.args = args

        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks

        self.obs_dim = len(obs_space)
        self.share_obs_dim = len(share_obs_space)

        self.action_dim = len(action_space)
        self.num_actions = 1

        self.num_agents = self.args.num_agents

        self.tpdv = dict(dtype=torch.float32, device=self.args.device)

        self.transformer = MultiAgentTransformer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            num_block=self.args.num_block,
            num_embd=self.args.num_embd,
            num_head=self.args.num_head,
            device=self.args.device,
            dec_actor=False,
            share_actor=False,
        )

        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(
        self,
        shared_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        deterministic=False,
    ):
        shared_obs = shared_obs.reshape(
            -1,
            self.num_agents,
            self.share_obs_dim,
        )
        obs = obs.reshape(
            -1,
            self.num_agents,
            self.obs_dim,
        )
        actions, action_log_probs, values = self.transformer.get_actions(
            obs=obs,
            available_actions=None,
            deterministic=deterministic,
        )
        actions = actions.view(-1, self.num_actions)
        action_log_probs = action_log_probs.view(-1, self.num_actions)
        values = values.view(-1, self.num_actions)

        # 使用されないが互換性のために
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)

        return actions, action_log_probs, values, rnn_states_actor, rnn_states_critic

    def get_values(
        self,
        shared_obs,
        obs,
    ):
        shared_obs = shared_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        values = self.transformer.get_values(obs=obs)
        values = values.view(-1, 1)
        return values

    def evaluate_actions(
        self,
        shared_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        shared_obs = shared_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.num_actions)  # [B, N, 1]

        if available_actions is not None:
            available_actions = available_actions.reshape(
                -1, self.num_agents, self.num_actions
            )

        action_log_probs, values, entropy = self.transformer(
            obs=obs,
            action=actions,
            available_actions=available_actions,
        )

        action_log_probs = action_log_probs.view(-1, self.num_actions)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.num_actions)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy

    def eval(
        self,
    ):
        self.transformer.eval()
