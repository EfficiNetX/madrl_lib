from runner.shared.base_runner import BaseRunner
import numpy as np
import torch
from typing import Tuple


class ValueMainRunner(BaseRunner):
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
            # 一時的に保持するTorchテンソル（学習用）
            obs_buf = torch.zeros(
                self.num_rollout_threads,
                self.episode_length + 1,
                self.num_agents,
                self.obs_dim,
                dtype=torch.float32,
                device=self.all_args.device,
            )
            rewards_buf = torch.zeros(
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
                1,
                dtype=torch.float32,
                device=self.all_args.device,
            )
            dones_buf = torch.zeros(
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
                dtype=torch.bool,
                device=self.all_args.device,
            )
            actions_buf = torch.zeros(
                self.num_rollout_threads,
                self.episode_length,
                self.num_agents,
                1,
                dtype=torch.int64,
                device=self.all_args.device,
            )
            mask_buf = torch.ones(
                self.num_rollout_threads,
                self.episode_length,
                dtype=torch.bool,
                device=self.all_args.device,
            )  # (num_rollout_threads, episode_length) (bool)
            avail_actions_buf = torch.ones(
                self.num_rollout_threads,
                self.episode_length + 1,
                self.num_agents,
                self.action_dim,
                dtype=torch.bool,
                device=self.all_args.device,
            )
            # 環境をリセットして初期観測を取得 && hidden stateの初期化
            obs = self.envs.reset()  # Numpy配列 (num_rollout_threads, num_agents, obs_dim)
            obs = torch.from_numpy(obs).float().to(self.all_args.device)
            hidden_states = self.policy.init_hidden(
                self.num_rollout_threads
            )  # Tensor (num_rollout_threads, num_agents, hidden_dim)
            # 1エピソードのデータ収集
            dones = torch.zeros(
                (self.num_rollout_threads, self.num_agents, 1),
                dtype=torch.bool,
            ).to(self.all_args.device)
            mask_buf[:, 0] = torch.zeros(
                (self.num_rollout_threads,), dtype=torch.bool
            ).to(
                self.all_args.device
            )  # エピソード開始直後はすべての環境が未終了なのでmaskはFalse
            obs_buf[:, 0] = obs
            avail_actions = self.envs.get_avail_actions()
            avail_actions = torch.from_numpy(avail_actions).bool().to(self.all_args.device)
            avail_actions_buf[:, 0] = avail_actions
            for step in range(self.episode_length):
                # obsをTensorに変換
                actions, next_hidden_states = self.collect(obs, hidden_states, dones)
                actions_buf[:, step] = actions
                actions = actions.cpu().numpy()  # (num_rollout_threads, num_agents, 1)
                action_list.append(actions[0])  # visualize用に保存
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
                if not step == self.episode_length - 1:  # 最終ステップではない場合
                    obs_list.append(next_obs[0])  # visualize用に保存
                reward_list.append(rewards[0])  # visualize用に保存
                next_obs = torch.from_numpy(next_obs).float().to(self.all_args.device)
                rewards = torch.from_numpy(rewards).float().to(self.all_args.device)
                dones = torch.from_numpy(dones).bool().to(self.all_args.device)
                obs_buf[:, step + 1] = next_obs
                avail_actions_buf[:, step + 1] = (
                    torch.from_numpy(self.envs.get_avail_actions()).bool().to(self.all_args.device)
                )
                rewards_buf[:, step] = rewards
                dones_buf[:, step] = dones
                if step + 1 < self.episode_length:
                    mask_buf[:, step + 1] = dones.all(dim=1)  # (num_rollout_threads,)
                t_env += self.num_rollout_threads - mask_buf[:, step].sum().item()
                obs = next_obs
                hidden_states = next_hidden_states

            # 各テンソルをnumpy配列に変換してselfに保持
            # share_obsは(obsのエージェント次元をフラット化)
            share_obs = obs_buf.reshape(
                self.num_rollout_threads, self.episode_length + 1, -1
            )
            self.share_obs = share_obs.detach().cpu().numpy().astype(np.float32)
            self.obs = obs_buf.detach().cpu().numpy().astype(np.float32)
            self.actions = actions_buf.detach().cpu().numpy().astype(np.int64)
            self.rewards = rewards_buf.detach().cpu().numpy().astype(np.float32)
            self.dones = dones_buf.detach().cpu().numpy().astype(bool)
            self.mask = mask_buf.detach().cpu().numpy().astype(bool)
            self.avail_actions = avail_actions_buf.detach().cpu().numpy().astype(bool)

            # バッファへ挿入（位置引数で渡す）
            self.insert()
            # バッチ数分のデータが溜まったら学習を行う
            if self.can_sample():
                episode_samples = self.sample()
                self.train(episode_samples)

            if episode % self.log_interval == 0:
                print(f"Episode {episode} finished.")
                print(f"finished step ratio: {t_env / self.num_env_steps}")

                self.visualizer(
                    episode=episode,
                    obs_list=obs_list,
                    reward_list=reward_list,
                    action_list=action_list,
                )

    def insert(self):
        # BufferのAPIに合わせて，selfに保持した各配列を渡す
        self.buffer.insert(
            self.share_obs,
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.mask,
            self.avail_actions,
        )

    def collect(self, obs, hidden_states, dones) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1ステップ分のデータ収集
        actions, next_hidden_states = self.policy.get_actions(obs, hidden_states, dones)
        return actions, next_hidden_states

    def warmup(self):
        # 環境をリセットして初期観測を取得
        obs = self.envs.reset()
        # 必要なら初期観測をバッファや変数に保存
        self.initial_obs = obs.copy()
        self.initial_hidden_states = self.policy.init_hidden(self.num_rollout_threads)

    def train(self, episode_samples):
        self.trainer.train(episode_samples)

    def update_epsilon(self, t_env):
        self.policy.update_epsilon(t_env)

    def can_sample(self):
        return self.buffer.can_sample(self.all_args.qmix_batch_size)

    def sample(self):
        return self.buffer.sample(self.all_args.qmix_batch_size)
