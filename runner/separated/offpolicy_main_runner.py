import time

import numpy as np
import torch

from runner.separated.base_runner import BaseRunner


class OffPolicyMainRunner(BaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)
        if self.all_args.algorithm_name == "HASAC":
            from algorithms.hasac.algorithm.hasac_policy import HASACPolicy as Policy

            if self.all_args.use_centralized_V:
                from algorithms.hasac.hasac_centralized_trainer import (
                    CentralizedSACTrainer as Trainer,
                )
            else:
                from algorithms.hasac.hasac_independent_trainer import (
                    IndependentSACTrainer as Trainer,
                )
            from utils.offpolicy_buffer import EpisodeReplayBuffer as Buffer
        self.policy = []
        self.obs_dim = len(self.envs.observation_space[0])
        # すべてのエージェントで同じ観測・行動次元を仮定
        self.action_shape = (
            1
            if self.all_args.action_type == "discrete"
            else len(self.envs.action_space[0])
        )
        self.action_dim = (
            len(self.envs.action_space[0])
            if self.all_args.action_type == "discrete"
            else len(self.envs.action_space[0])
        )
        for agent_id in range(self.all_args.num_agents):
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                self.share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.policy.append(po)
            # algorithm
        self.trainer = Trainer(
            args=self.all_args,
            policy=self.policy,
        )
        self.buffer = Buffer(
            args=self.all_args,
            num_agents=self.all_args.num_agents,
            obs_space=self.envs.observation_space[0],
            share_obs_space=self.share_observation_space,
            action_space=self.envs.action_space[0],
            action_shape=self.action_shape,
        )
        self.show_reward_list = []
        print("obs_space", self.envs.observation_space[0])
        print("share_obs_space", self.share_observation_space)
        print("action_space", self.envs.action_space[0])

    def run(self) -> None:
        t_env = 0  # 環境ステップ数のカウンタ
        start = time.time()
        episodes = (
            int(self.all_args.num_env_steps)
            // self.all_args.episode_length
            // self.all_args.num_rollout_threads
        )
        for episode in range(episodes):
            # visualize用にデータを保存するリストを定義
            obs_list, reward_list, action_list = [], [], []
            # エピソード保存用のバッファ
            self.warmup()
            obs = self.initial_obs  # numpy
            # 初期donesはFalse
            dones = np.zeros(
                (self.all_args.num_rollout_threads, self.all_args.num_agents),
                dtype=bool,
            )
            for step in range(self.all_args.episode_length):
                actions, actions_env = self.collect(obs)
                obs, rewards, dones = self.envs.step(actions_env)
                if step != self.all_args.episode_length - 1:
                    obs_list.append(obs[0])
                action_list.append(actions[0])
                reward_list.append(rewards[0])
                # 1ステップ分のデータをトラジェクトリバッファに保存
                self.save_step(
                    obs,
                    actions,
                    rewards,
                    dones,
                    self.envs.get_avail_actions(),
                    step,
                )
                t_env += (~self.mask_trajectory_buffer)[:, step].sum()

            # バッファへ挿入（位置引数で渡す）
            self.insert(
                shared_obs=self.obs_trajectory_buffer.reshape(
                    self.all_args.num_rollout_threads,
                    self.all_args.episode_length + 1,
                    -1,
                ),
                obs=self.obs_trajectory_buffer,
                actions=self.actions_trajectory_buffer,
                rewards=self.rewards_trajectory_buffer,
                dones=self.dones_trajectory_buffer,
                mask=self.mask_trajectory_buffer,
                avail_actions=self.avail_actions_trajectory_buffer,
            )
            # 後ろ２つの次元を合計して平均報酬を計算
            # 1エピソードあたりの合計報酬を計算し、スレッド間で平均を取る
            episode_rewards = np.sum(self.rewards_trajectory_buffer, axis=(1, 2, 3))
            self.show_reward_list.append(np.mean(episode_rewards))

            if len(self.show_reward_list) == 10:
                print(
                    "Episode {} Average Reward: {}".format(
                        episode,
                        np.mean(self.show_reward_list),
                    )
                )
                print("Min Reward: {}".format(np.min(self.show_reward_list)))
                print("Max Reward: {}".format(np.max(self.show_reward_list)))
                self.show_reward_list = []
                # 一番最初のstateを入力したときのactionの確率分布を表示
                for agent_id in range(self.all_args.num_agents):
                    obs_t = torch.tensor(
                        self.obs_trajectory_buffer[0, 0, agent_id, :],
                        device=self.all_args.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        _, _, action_probs = self.policy[
                            agent_id
                        ].get_action_with_probability(obs_t)
                    print(
                        "Agent {} Action Probabilities: {}".format(
                            agent_id,
                            action_probs.cpu().detach().numpy(),
                        )
                    )
            # バッチ数分のデータが溜まったら学習を行う
            if (
                self.buffer.can_sample(self.all_args.batch_size)
                and episode > self.all_args.warmup_episodes
            ):
                if episode % self.all_args.train_interval != 0:
                    continue
                episode_samples = self.buffer.sample(self.all_args.batch_size)
                self.trainer.train(episode_samples)

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
                            np.mean(self.rewards_trajectory_buffer[:, :, agent_id, :])
                            * self.all_args.episode_length,
                        )
                    )
                self.visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

    def save_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        avail_actions: np.ndarray,
        step: int,
    ) -> None:
        self.obs_trajectory_buffer[:, step + 1] = obs.copy()
        self.actions_trajectory_buffer[:, step] = actions.copy()
        self.rewards_trajectory_buffer[:, step] = rewards.copy()
        self.dones_trajectory_buffer[:, step] = dones.copy()
        self.avail_actions_trajectory_buffer[:, step + 1] = avail_actions.copy()
        if step + 1 < self.all_args.episode_length:
            # 全エージェント終了でTrue
            self.mask_trajectory_buffer[:, step + 1] = dones.all(axis=1)

    def insert(
        self,
        shared_obs: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        mask: np.ndarray,
        avail_actions: np.ndarray,
    ) -> None:
        # それぞれのエージェントのバッファにデータを挿入
        self.buffer.insert(
            shared_obs,
            obs,
            actions,
            rewards,
            dones,
            mask,
            avail_actions,
        )

    @torch.no_grad()
    def collect(self, obs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        各エージェントの行動を取得

        Args:
            obs_np: (num_rollout_threads, num_agents, obs_dim)

        Returns:
            actions_np: (num_rollout_threads, num_agents, 1)
            actions_env: (num_rollout_threads, num_agents, action_dim)
        """
        obs_t = torch.from_numpy(obs_np).float().to(self.all_args.device)

        # 各エージェントの行動を取得
        actions = []
        actions_env = []
        for agent_id in range(self.all_args.num_agents):
            action, action_env = self.policy[agent_id].get_action(
                obs=obs_t[:, agent_id, :],
                deterministic=self.all_args.deterministic,
            )
            actions.append(action)
            actions_env.append(action_env)
        # スタック: (num_rollout_threads, num_agents)
        actions = torch.stack(actions, dim=1)
        actions_env = torch.stack(actions_env, dim=1)
        # numpy変換
        actions = actions.cpu().numpy()
        actions_env = actions_env.cpu().numpy()
        return actions, actions_env

    def warmup(self):
        # 環境をリセットして初期観測を取得
        obs = self.envs.reset()
        # 必要なら初期観測をバッファや変数に保存
        self.initial_obs = obs.copy()
        self.obs_trajectory_buffer = np.zeros(
            (
                self.all_args.num_rollout_threads,
                self.all_args.episode_length + 1,
                self.all_args.num_agents,
                self.obs_dim,
            ),
            dtype=np.float32,
        )
        self.rewards_trajectory_buffer = np.zeros(
            (
                self.all_args.num_rollout_threads,
                self.all_args.episode_length,
                self.all_args.num_agents,
                1,
            ),
            dtype=np.float32,
        )
        self.dones_trajectory_buffer = np.zeros(
            (
                self.all_args.num_rollout_threads,
                self.all_args.episode_length,
                self.all_args.num_agents,
            ),
            dtype=bool,
        )
        self.actions_trajectory_buffer = np.zeros(
            (
                self.all_args.num_rollout_threads,
                self.all_args.episode_length,
                self.all_args.num_agents,
                self.action_shape,
            ),
            dtype=np.int64,
        )
        self.mask_trajectory_buffer = np.ones(
            (self.all_args.num_rollout_threads, self.all_args.episode_length),
            dtype=bool,
        )
        self.avail_actions_trajectory_buffer = np.ones(
            (
                self.all_args.num_rollout_threads,
                self.all_args.episode_length + 1,
                self.all_args.num_agents,
                self.action_dim,
            ),
            dtype=bool,
        )
        self.mask_trajectory_buffer[:, 0] = False  # エピソード開始直後は全環境未終了
        self.obs_trajectory_buffer[:, 0] = self.initial_obs  # numpy
        self.avail_actions_trajectory_buffer[:, 0] = (
            self.envs.get_avail_actions().copy()
        )
