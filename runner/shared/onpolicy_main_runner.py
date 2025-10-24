from tracemalloc import start
from runner.shared.base_runner import BaseRunner
import numpy as np
import torch
import time


class ValueMainRunner(BaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self) -> None:
        t_env = 0  # 環境ステップ数のカウンタ
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.num_rollout_threads
        for episode in range(episodes):
            # epsilonの線形減衰
            self.trainer.policy.update_epsilon(t_env)
            # visualize用にデータを保存するリストを定義
            obs_list, reward_list, action_list = [], [], []
            # エピソード保存用のバッファ
            obs_buf = np.zeros(
                (
                    self.num_rollout_threads,
                    self.episode_length + 1,
                    self.num_agents,
                    self.obs_dim,
                ),
                dtype=np.float32,
            )
            rewards_buf = np.zeros(
                (
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )
            dones_buf = np.zeros(
                (
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                ),
                dtype=bool,
            )
            actions_buf = np.zeros(
                (
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                    1,
                ),
                dtype=np.int64,
            )
            mask_buf = np.ones(
                (self.num_rollout_threads, self.episode_length),
                dtype=bool,
            )
            avail_actions_buf = np.ones(
                (
                    self.num_rollout_threads,
                    self.episode_length + 1,
                    self.num_agents,
                    self.action_dim,
                ),
                dtype=bool,
            )
            self.warmup()
            mask_buf[:, 0] = False  # エピソード開始直後は全環境未終了
            obs_buf[:, 0] = self.initial_obs  # numpy
            avail_actions_buf[:, 0] = self.envs.get_avail_actions()
            obs = self.initial_obs  # numpy
            # 初期donesはFalse
            dones = np.zeros((self.num_rollout_threads, self.num_agents), dtype=bool)
            for step in range(self.episode_length):
                # 方策はtorch入力を想定、この時点のobs/donesだけ最小限に変換
                actions, actions_env = self.collect(obs, dones)
                actions_buf[:, step] = actions
                obs, rewards, dones = self.envs.step(actions_env)
                if step != self.episode_length - 1:
                    obs_list.append(obs[0])
                action_list.append(actions[0])
                reward_list.append(rewards[0])
                obs_buf[:, step + 1] = obs.astype(np.float32)
                avail_actions_buf[:, step + 1] = self.envs.get_avail_actions()
                rewards_buf[:, step] = rewards.astype(np.float32)
                dones_buf[:, step] = dones.astype(bool)
                if step + 1 < self.episode_length:
                    # 全エージェント終了でTrue
                    mask_buf[:, step + 1] = dones.all(axis=1)

            # バッファへ挿入（位置引数で渡す）
            self.insert(
                shared_obs=obs_buf.reshape(self.num_rollout_threads, self.episode_length + 1, -1),
                obs=obs_buf,
                actions=actions_buf,
                rewards=rewards_buf,
                dones=dones_buf,
                mask=mask_buf,
                avail_actions=avail_actions_buf,
            )
            # バッチ数分のデータが溜まったら学習を行う
            if self.buffer.can_sample(self.all_args.batch_size):
                episode_samples = self.buffer.sample(self.all_args.batch_size)
                self.trainer.train(episode_samples)

            total_num_steps = (episode + 1) * self.episode_length * self.num_rollout_threads

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
                            np.mean(rewards_buf[:, :, agent_id, :]) * self.episode_length,
                        )
                    )
                self.visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

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

    def collect(self, obs_np: np.ndarray, dones_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
