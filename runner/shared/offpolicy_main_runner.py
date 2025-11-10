import time

import numpy as np
import torch

from runner.shared.base_runner import BaseRunner


class OnPolicyMainRunner(BaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self) -> None:
        t_env = 0  # 環境ステップ数のカウンタ
        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.num_rollout_threads
        )
        for episode in range(episodes):
            # epsilonの線形減衰
            self.trainer.policy.update_epsilon(t_env)
            # visualize用にデータを保存するリストを定義
            obs_list, reward_list, action_list = [], [], []
            # エピソード保存用のバッファ
            self.warmup()
            obs = self.initial_obs  # numpy
            # 初期donesはFalse
            dones = np.zeros((self.num_rollout_threads, self.num_agents), dtype=bool)
            for step in range(self.episode_length):
                actions, actions_env = self.collect(obs, dones)
                obs, rewards, dones = self.envs.step(actions_env)
                if step != self.episode_length - 1:
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
                    self.num_rollout_threads, self.episode_length + 1, -1
                ),
                obs=self.obs_trajectory_buffer,
                actions=self.actions_trajectory_buffer,
                rewards=self.rewards_trajectory_buffer,
                dones=self.dones_trajectory_buffer,
                mask=self.mask_trajectory_buffer,
                avail_actions=self.avail_actions_trajectory_buffer,
            )
            # バッチ数分のデータが溜まったら学習を行う
            if self.buffer.can_sample(self.all_args.batch_size):
                episode_samples = self.buffer.sample(self.all_args.batch_size)
                self.trainer.train(episode_samples)

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
                            np.mean(self.rewards_trajectory_buffer[:, :, agent_id, :])
                            * self.episode_length,
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
        if step + 1 < self.episode_length:
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
    def collect(
        self, obs_np: np.ndarray, dones_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # numpy -> torch（最小限の変換のみ）
        obs_t = torch.from_numpy(obs_np).float().to(self.all_args.device)
        dones_t = torch.from_numpy(dones_np).bool().to(self.all_args.device)
        actions = self.trainer.policy.get_actions(obs_t, dones_t, deterministic=False)
        # actionsをnumpyに変換
        actions = actions.cpu().numpy().astype(np.int64)
        # one-hot化（numpy）
        actions_env = np.eye(
            self.action_dim,
            dtype=np.float32,
        )[actions.squeeze(-1)]
        return actions, actions_env

    def warmup(self):
        # 環境をリセットして初期観測を取得
        obs = self.envs.reset()
        # 必要なら初期観測をバッファや変数に保存
        self.initial_obs = obs.copy()
        self.policy.init_hidden(batch_size=self.num_rollout_threads)
        self.obs_trajectory_buffer = np.zeros(
            (
                self.num_rollout_threads,
                self.episode_length + 1,
                self.num_agents,
                self.obs_dim,
            ),
            dtype=np.float32,
        )
        self.rewards_trajectory_buffer = np.zeros(
            (
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )
        self.dones_trajectory_buffer = np.zeros(
            (
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
            ),
            dtype=bool,
        )
        self.actions_trajectory_buffer = np.zeros(
            (
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
                1,
            ),
            dtype=np.int64,
        )
        self.mask_trajectory_buffer = np.ones(
            (self.num_rollout_threads, self.episode_length),
            dtype=bool,
        )
        self.avail_actions_trajectory_buffer = np.ones(
            (
                self.num_rollout_threads,
                self.episode_length + 1,
                self.num_agents,
                self.action_dim,
            ),
            dtype=bool,
        )
        self.mask_trajectory_buffer[:, 0] = False  # エピソード開始直後は全環境未終了
        self.obs_trajectory_buffer[:, 0] = self.initial_obs  # numpy
        self.avail_actions_trajectory_buffer[:, 0] = (
            self.envs.get_avail_actions().copy()
        )
