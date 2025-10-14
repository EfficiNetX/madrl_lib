import time
from runner.shared.value_base_runner import ValueBaseRunner
import numpy as np
import torch


class ValueMainRunner(ValueBaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        t_env = 0  # 環境ステップ数のカウンタ
        episode = 0  # エピソード数のカウンタ
        while t_env < self.num_env_steps:
            episode += self.num_rollout_threads
            # epsilonの線形減衰
            self.trainer.policy.update_epsilon(t_env)
            # visualize用にデータを保存するリストを定義
            obs_list, reward_list, action_list = [], [], []
            episode_data = {
                "obs": torch.zeros(
                    self.num_rollout_threads,
                    self.episode_length + 1,
                    self.num_agents,
                    self.obs_dim,
                    dtype=torch.float32,
                ),
                "rewards": torch.zeros(
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                    1,
                    dtype=torch.float32,
                ),
                "dones": torch.zeros(
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                    dtype=torch.bool,
                ),
                "actions": torch.zeros(
                    self.num_rollout_threads,
                    self.episode_length,
                    self.num_agents,
                    1,
                    dtype=torch.int64,
                ),
                "mask": torch.ones(
                    self.num_rollout_threads, self.episode_length, dtype=torch.bool
                ),  # (num_rollout_threads, episode_length) (bool)
            }
            # 環境をリセットして初期観測を取得 && hidden stateの初期化
            obs = self.envs.reset()  # Numpy配列 (num_rollout_threads, num_agents, obs_dim)
            obs = torch.from_numpy(obs).float().to(self.all_args.device)
            hidden_states = self.policy.init_hidden(
                self.num_rollout_threads
            )  # Tensor (num_rollout_threads, num_agents, hidden_dim)
            # 1エピソードのデータ収集
            dones = torch.zeros(
                (self.num_rollout_threads, self.num_agents, 1), dtype=torch.bool
            ).to(self.all_args.device)
            episode_data["mask"][:, 0] = torch.zeros(
                (self.num_rollout_threads,), dtype=torch.bool
            ).to(
                self.all_args.device
            )  # エピソード開始直後はすべての環境が未終了なのでmaskはFalse
            for step in range(self.episode_length):
                episode_data["obs"][:, step] = obs
                obs_list.append(obs[0].cpu().numpy())  # visualize用に保存
                # obsをTensorに変換
                actions, next_hidden_states = self.collect(obs, hidden_states, dones)
                episode_data["actions"][:, step] = actions
                actions = actions.cpu().numpy()  # (num_rollout_threads, num_agents, 1)
                # one-hot化
                num_actions = (
                    self.envs.action_space[0].__len__()
                    if hasattr(self.envs.action_space[0], "__len__")
                    else len(self.envs.action_space[0])
                )
                actions_onehot = np.eye(num_actions)[
                    actions.squeeze(-1)
                ]  # (num_rollout_threads, num_agents, num_actions)
                next_obs, rewards, dones = self.envs.step(
                    actions_onehot
                )  # 入力値、返り値はNumpy配列
                reward_list.append(rewards[0])  # visualize用に保存
                action_list.append(actions[0])  # visualize用に保存
                next_obs = torch.from_numpy(next_obs).float().to(self.all_args.device)
                rewards = torch.from_numpy(rewards).float().to(self.all_args.device)
                dones = torch.from_numpy(dones).bool().to(self.all_args.device)
                episode_data["rewards"][:, step] = rewards
                if step + 1 < self.episode_length:
                    episode_data["mask"][:, step + 1] = dones.all(
                        dim=1
                    )  # (           num_rollout_threads,)
                t_env += self.num_rollout_threads - episode_data["mask"][:, step].sum().item()

                obs = next_obs
                hidden_states = next_hidden_states
                if dones.all():  # すべての環境が終了したらエピソード終了
                    break

            # ループ終了後の最後の状態を保存
            episode_data["obs"][:, step + 1] = obs

            self.insert(episode_data)
            # バッチ数分のデータが溜まったら学習を行う
            if self.can_sample():
                episode_samples = self.sample()
                self.train(episode_samples)

            if episode * 10 % self.log_interval == 0:
                print(f"Step {t_env}/{self.num_env_steps}")

            if episode % self.log_interval == 0:
                print(f"Episode {episode}")

                self.visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

    def insert(self, episode_data):
        # obsからshare_obsを取得してepisode_dataに追加
        # TODO: 必要ならば，obsを共有しない場合の分岐を実装する
        share_obs = episode_data["obs"]
        share_obs = share_obs.reshape(self.num_rollout_threads, self.episode_length + 1, -1)
        episode_data["share_obs"] = share_obs
        self.buffer.insert(episode_data)
