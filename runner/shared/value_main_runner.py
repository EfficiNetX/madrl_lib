import time
from runner.shared.value_base_runner import ValueBaseRunner
import numpy as np


class ValueMainRunner(ValueBaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.num_rollout_threads
        )

        for episode in range(episodes):
            # TODO : 学習率の減衰
            """
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            """
            # バッファに保存するための辞書
            # TODO: episode_dataのクラス化
            # TODO: episode_lengthよりも早く終わった場合の処理
            episode_data = {
                "obs": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length + 1,
                        self.n_agents,
                        self.obs_dim,
                    ),
                    dtype=np.float32,
                ),
                "rewards": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                    ),
                    dtype=np.float32,
                ),
                "dones": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                    ),
                    dtype=bool,
                ),
                "actions": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        1,
                    ),
                    dtype=np.int,
                ),
                "filled": np.zeros(
                    (self.num_rollout_threads, self.episode_length),
                    dtype=bool,
                ),
            }
            # 環境をリセットして初期観測を取得 && hidden stateの初期化
            obs = self.envs.reset()
            hidden_states = self.policy.init_hidden(self.num_rollout_threads)
            # 1エピソードのデータ収集
            dones = np.zeros((self.num_rollout_threads, self.n_agents, 1))
            for step in range(self.episode_length):
                episode_data["obs"][:, step] = obs
                actions, next_hidden_states = self.collect(
                    obs, hidden_states, dones
                )
                episode_data["actions"][:, step] = actions
                next_obs, rewards, dones, info = self.envs.step(actions)
                episode_data["rewards"][:, step, 0] = rewards
                episode_data["dones"][:, step] = dones
                episode_data["filled"][:, step, 0] = True

                obs = next_obs
                hidden_states = next_hidden_states
                if dones.any():
                    break

            # ループ終了後の最後の状態を保存
            episode_data["obs"][:, step + 1] = obs

            self.insert(episode_data)
            # バッチ数分のデータが溜まったら学習を行う
            if self.can_sample():
                episode_samples = self.sample()
                self.train(episode_samples)

            if episode % self.log_interval == 0:
                print(f"Episode {episode}/{episodes}")

    def insert(self, episode_data):
        # obsからshare_obsを取得してepisode_dataに追加
        # TODO: 必要ならば，obsを共有しない場合の分岐を実装する
        share_obs = episode_data["obs"]
        share_obs = share_obs.reshape(
            self.num_rollout_threads, self.episode_length + 1, -1
        )
        episode_data["share_obs"] = share_obs
        self.buffer.insert(episode_data)
