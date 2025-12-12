import torch

from algorithms.r_mappo.algorithm.r_actor_critc import R_Actor, R_Critic


class HAPPO_Policy:
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        action_space,
    ):
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space

        self.actor = R_Actor(
            args=args,
            obs_space=self.obs_space,
            action_space=self.action_space,
        )

        ######################################Please Note#########################################
        #####   We create one critic for each agent, but they are trained with same data     #####
        #####   and using same update setting. Therefore they have the same parameter,       #####
        #####   you can regard them as the same critic.                                      #####
        ##########################################################################################
        self.critic = R_Critic(
            args=args,
            share_obs_space=self.share_obs_space,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_actions(
        self,
        shared_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=None,
    ):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        values, rnn_states_critic = self.critic(
            shared_obs,
            rnn_states_critic,
            masks,
        )
        return actions, action_log_probs, values, rnn_states_actor, rnn_states_critic

    def get_values(self, shared_obs, rnn_states_critic, masks):
        values, _ = self.critic(
            shared_obs,
            rnn_states_critic,
            masks,
        )
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
    ):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs=obs,
            rnn_states=rnn_states_actor,
            action=actions,
            masks=masks,
            available_actions=available_actions,
        )
        values, _ = self.critic(
            shared_obs=shared_obs,
            rnn_states=rnn_states_critic,
            masks=masks,
        )
        return values, action_log_probs, dist_entropy
