import importlib
import time
from itertools import chain

import numpy as np
import torch

from envs.DemoUser.DemoUser_visualize import visualizer
from runner.separated.base_runner import BaseRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class UserEnvRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        # visualizerのimport
        user_name = config["args"].user_name
        visualizeClass = importlib.import_module(
            f"envs.{user_name}.{user_name}_visualize"
        )
        self.visualizer = getattr(visualizeClass, "visualizer")

    def run(self):
        self.warmup()
        start = time.time()

        episodes = (
            int(self.all_args.num_env_steps)
            // self.all_args.episode_length
            // self.all_args.num_rollout_threads
        )
        for episode in range(episodes):
            print("episode ={}".format(episode))
            if self.all_args.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            obs_list, reward_list, action_list = [], [], []
            for step in range(self.all_args.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                obs, rewards, dones = self.envs.step(actions_env)
                data = (
                    obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                # insert data into buffer
                self.insert(data)

                # visualizeのためにデータを保存
                if (
                    step != self.all_args.episode_length - 1
                ):  # 最終ステップの次状態は可視化フレーム外に出るため保存しない
                    obs_list.append(obs[0])
                reward_list.append(rewards[0])
                action_list.append(actions[0])

            # compute return and update network
            self.compute()
            _ = self.train()
            # post process
            total_num_steps = (
                (episode + 1)
                * self.all_args.episode_length
                * self.all_args.num_rollout_threads
            )
            if episode % self.all_args.log_interval == 0:
                print(
                    "Scenario {} Algo {} updates {}/{} episodes,"
                    " total num timesteps {}/{}, FPS {}.".format(
                        self.all_args.user_name,
                        self.all_args.algorithm_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.all_args.num_env_steps,
                        int(total_num_steps / (time.time() - start)),
                    )
                )
                for agent_id in range(self.all_args.num_agents):
                    print(
                        "agent {}: average episode rewards is {}".format(
                            agent_id,
                            np.mean(self.buffer[agent_id].rewards)
                            * self.all_args.episode_length,
                        )
                    )

                visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

    def warmup(self):
        # reset the env
        obs = self.envs.reset()
        share_obs = []

        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.all_args.num_agents):
            if not self.all_args.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.all_args.num_agents):
            self.trainer[agent_id].prep_rollout()
            action, action_log_prob, value, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )

            values.append(_t2n(value))
            action = _t2n(action)

            # rearrange action shape
            num_actions = len(self.envs.action_space[0])
            action_env = np.squeeze(
                np.eye(num_actions)[action], 1
            )  # [B, 1, num_actions] -> [B, num_actions]

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.all_args.num_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        rnn_states[dones] = np.zeros(
            (dones.sum(), self.all_args.recurrent_N, self.all_args.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones] = np.zeros(
            (dones.sum(), self.all_args.recurrent_N, self.all_args.hidden_size)
        )
        masks = np.ones(
            (self.all_args.num_rollout_threads, self.all_args.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.all_args.num_agents):
            if not self.all_args.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].insert(
                share_obs=share_obs,
                obs=np.array(list(obs[:, agent_id])),
                rnn_states_actor=rnn_states[:, agent_id],
                rnn_states_critic=rnn_states_critic[:, agent_id],
                actions=actions[:, agent_id],
                action_log_probs=action_log_probs[:, agent_id],
                value_preds=values[:, agent_id],
                rewards=rewards[:, agent_id],
                masks=masks[:, agent_id],
            )
