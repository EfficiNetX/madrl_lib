import time
from runner.separated.value_base_runner import ValueBaseRunner
import numpy as np


class ValueMainRunner(ValueBaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.num_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            # TODO: 環境を並列で実行できるようにする
            episode_data = {
                "state": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        self.state_dim,
                    ),
                    dtype=np.float32,
                ),
                "obs": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        self.obs_space,
                    ),
                    dtype=np.float32,
                ),
                "actions": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        self.action_space,
                    ),
                    dtype=np.float32,
                ),
                "rewards": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        1,
                    ),
                    dtype=np.float32,
                ),
                "dones": np.zeros(
                    (
                        self.num_rollout_threads,
                        self.episode_length,
                        self.n_agents,
                        1,
                    ),
                    dtype=bool,
                ),
                "filled": np.zeros(
                    (self.num_rollout_threads, self.episode_length, 1),
                    dtype=int,
                ),
            }  # バッファに保存するための辞書
            obs = (
                self.envs.reset()
            )  # (self.num_rollout_threads, self.obs_space)
            hidden_states = self.policy.init_hidden(
                batch_size=self.num_rollout_threads
            )
            # for文: 1エピソードのデータ収集を行う
            for step in range(self.episode_length):
                # TODO: デバッグ時に保存データの次元があってるか確認
                episode_data["obs"][:, step] = obs
                actions, next_hidden_states = self.policy.select_actions(
                    obs, hidden_states
                )
                episode_data["actions"][:, step] = actions
                next_obs, rewards, dones, info = self.envs.step(actions)
                episode_data["state"][:, step] = info["state"]
                episode_data["rewards"][:, step] = rewards
                episode_data["dones"][:, step] = dones
                episode_data["filled"][:, step] = 1

                # state, avail_actions, filledも同様に保存
                obs = next_obs
                hidden_states = next_hidden_states
                if dones.any():
                    break
            # TODO: 終了時の情報も保存する

            self.insert(episode_data)
            # バッチ数分のデータが溜まったら学習を行う
            if self.can_sample():
                episode_samples = self.sample(self.all_args.qmix_batch_size)
                self.train(episode_samples)

            if episode % self.log_interval == 0:
                print(f"Episode {episode}/{episodes}")
