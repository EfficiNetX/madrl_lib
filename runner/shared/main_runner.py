import time

import numpy as np
import torch

from runner.shared.base_runner import BaseRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class UserEnvRunner(BaseRunner):
    """訓練・評価をするためのクラス"""

    def __init__(self, config):
        super().__init__(config)

    def run(
        self,
    ):
        self.warmup()
        start = time.time()

        episodes = (
            int(self.num_env_steps) // self.episode_length // self.num_rollout_threads
        )
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            obs_list, reward_list, action_list = [], [], []
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,  # [B, N, action_dim]
                ) = self.collect(step)

                # 報酬と次状態を観測
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
                # insert data
                self.insert(data)
                # visualizeのためにデータを保存
                # 最終ステップの次状態は可視化フレーム外に出るため保存しない
                if not step == self.episode_length - 1:
                    obs_list.append(obs[0])
                    print("obs_list length:", len(obs_list))
                reward_list.append(rewards[0])
                action_list.append(actions[0])

            # compute return and update network
            self.compute()
            _ = self.train()
            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.num_rollout_threads
            )

            if episode % self.log_interval == 0:
                print(
                    "Scenario {} Algo {} updates {}/{} episodes,"
                    " total num timesteps {}/{}, FPS {}.".format(
                        self.all_args.user_name,
                        self.algorithm_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (time.time() - start)),
                    )
                )

                for agent_id in range(self.num_agents):
                    print(
                        "agent {}: average episode rewards is {}".format(
                            agent_id,
                            np.mean(self.buffer.rewards[:, :, agent_id, :])
                            * self.episode_length,
                        )
                    )
                self.visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

    def warmup(self):
        # envをresetする
        obs = self.envs.reset()  # [envs, agents, obs_dim]
        if self.use_centralized_V:
            share_obs = obs.reshape(self.num_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, axis=1).repeat(
                self.num_agents,
                axis=1,
            )
        else:
            share_obs = obs
        # bufferにshare_obsとobsを格納する
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        action, action_log_prob, value, rnn_states_actor, rnn_states_critic = (
            self.trainer.policy.get_actions(
                shared_obs=np.concatenate(self.buffer.share_obs[step]),
                obs=np.concatenate(self.buffer.obs[step]),
                rnn_states_actor=np.concatenate(self.buffer.rnn_states[step]),
                rnn_states_critic=np.concatenate(self.buffer.rnn_states_critic[step]),
                masks=np.concatenate(self.buffer.masks[step]),
            )
        )

        # (self.envs, agents, dim)
        values = np.array(np.split(_t2n(value), self.num_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.num_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.num_rollout_threads)
        )
        rnn_states_actor = np.array(
            np.split(_t2n(rnn_states_actor), self.num_rollout_threads)
        )
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.num_rollout_threads)
        )

        # rearange action
        num_actions = len(self.envs.action_space[0])
        actions_env = np.eye(num_actions)[
            actions
        ]  # [B, N, 1] -> [B, N, 1, num_actions]
        actions_env = np.squeeze(actions_env, 2)  # 余計な次元を削除

        return (
            values,
            actions,
            action_log_probs,
            rnn_states_actor,
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
            (dones.sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones] = np.zeros(
            (dones.sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones(
            (self.num_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.num_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )
