import torch
import numpy as np


class ValueBaseRunner(object):
    """
    Value-basedアルゴリズム（QMIX/VDNなど）用のベースクラス
    """

    def __init__(self, config):
        self.all_args = config["args"]
        self.envs = config["envs"]
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.num_rollout_threads = self.all_args.num_rollout_threads
        self.n_agents = self.all_args.num_agents
        self.log_interval = self.all_args.log_interval
        self.algorithm_name = self.all_args.algorithm_name

        # Policy, Mixer, Trainer, Bufferの初期化（アルゴリズムごとに分岐）
        if self.algorithm_name == "QMIX":
            from algorithms.qmix.algorithm.qmix_policy import (
                QMIXPolicy as Policy,
            )
            from algorithms.qmix.algorithm.mixing_nn import QMixer as Mixer
            from algorithms.qmix.qmix_trainer import QMIXTrainer as Trainer
            from utils.shared_episode_buffer import (
                EpisodeReplayBuffer as Buffer,
            )
        elif self.algorithm_name == "VDN":
            from algorithms.vdn.vdn_policy import VDNPolicy as Policy
            from algorithms.vdn.mixing_nn import VDNMixer as Mixer
            from algorithms.vdn.vdn_trainer import VDNTrainer as Trainer
            from utils.shared_episode_buffer import (
                EpisodeReplayBuffer as Buffer,
            )
        else:
            raise NotImplementedError("Unknown value-based algorithm.")

        self.policy = Policy(self.all_args, ...)
        self.mixer = Mixer(self.all_args)
        self.trainer = Trainer(self.policy, self.mixer, self.all_args)
        self.buffer = Buffer(self.all_args, ...)

    def warmup(self):
        # 環境の初期化や必要な初期処理
        pass

    def insert(self, episode_data):
        # 1エピソード分のデータをバッファに追加
        self.buffer.add(episode_data)

    def train(self, t_env, episode_num):
        # バッファからサンプルして学習
        if self.buffer.can_sample(self.all_args.qmix_batch_size):
            batch = self.buffer.sample(self.all_args.qmix_batch_size)
            self.trainer.train(batch, t_env=t_env, episode_num=episode_num)

    def update_epsilon(self, t_env):
        # 探索率の減衰
        self.policy.update_epsilon(t_env)
