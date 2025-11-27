import importlib

import numpy as np
import torch


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class BaseRunner(object):
    """MARLアルゴリズムを訓練するためのベースクラス"""

    def __init__(self, config):
        self.all_args = config["args"]
        self.envs = config["envs"]

        obs_space = self.envs.observation_space[0]  # エージェント0の観測
        action_space = self.envs.action_space[0]
        print(obs_space, action_space)

        # interval
        self.all_args.log_interval = self.all_args.log_interval

        if self.all_args.algorithm_name == "MAT":
            share_obs_space = (
                self.envs.share_observation_space[0]
                if self.all_args.use_centralized_V
                else self.envs.observation_space[0]
            )  # エージェント0のshared観測
            # policy network
            from algorithms.mat.algorithm.transformer_policy import (
                TransformerPolicy as Policy,
            )
            from algorithms.mat.mat_trainer import MATTrainer as Trainer
            from utils.shared_buffer import ReplayBuffer as ReplayBuffer

            self.policy = Policy(
                args=self.all_args,
                obs_space=obs_space,
                share_obs_space=share_obs_space,
                action_space=action_space,
            )
            self.trainer = Trainer(
                args=self.all_args,
                policy=self.policy,
            )
        elif (
            self.all_args.algorithm_name == "RMAPPO"
            or self.all_args.algorithm_name == "IPPO"
        ):
            share_obs_space = (
                self.envs.share_observation_space[0]
                if self.all_args.use_centralized_V
                else self.envs.observation_space[0]
            )  # エージェント0のshared観測
            from algorithms.r_mappo.algorithm.RMAPPOPolicy import (
                RMAPPOPolicy as Policy,
            )
            from algorithms.r_mappo.rmappo_trainer import (
                RMAPPOTrainer as Trainer,
            )
            from utils.shared_buffer import ReplayBuffer as ReplayBuffer

            self.policy = Policy(
                args=self.all_args,
                obs_space=obs_space,
                share_obs_space=share_obs_space,
                action_space=action_space,
            )
            self.trainer = Trainer(
                args=self.all_args,
                policy=self.policy,
            )
        elif (
            self.all_args.algorithm_name == "QMIX"
            or self.all_args.algorithm_name == "VDN"
        ):
            share_obs_space = self.envs.share_observation_space[
                0
            ]  # QMIX/VDNではshared_obs_spaceを必ず使う
            from algorithms.valuedecomposition.algorithm.QPolicy import (
                QPolicy as Policy,
            )
            from algorithms.valuedecomposition.QTrainer import (
                QTrainer as Trainer,
            )
            from utils.offpolicy_buffer import (
                EpisodeReplayBuffer as ReplayBuffer,
            )

            if self.all_args.algorithm_name == "QMIX":
                from algorithms.valuedecomposition.algorithm.mixing_nn import (
                    QMixer as Mixer,
                )

                self.mixer = Mixer(
                    self.all_args,
                    obs_space,
                    share_obs_space,
                    action_space,
                )
            elif self.all_args.algorithm_name == "VDN":
                self.mixer = None  # VDNではミキサーは使用しない

            self.policy = Policy(
                args=self.all_args,
                obs_space=obs_space,
                share_obs_space=share_obs_space,
                action_space=action_space,
            )
            self.trainer = Trainer(
                args=self.all_args,
                policy=self.policy,
                mixer=self.mixer,
            )

        # buffer
        self.buffer = ReplayBuffer(
            args=self.all_args,
            num_agents=self.all_args.num_agents,
            obs_space=obs_space,
            share_obs_space=share_obs_space,
            action_space=action_space,
        )
        self.obs_dim = len(obs_space)
        self.action_dim = len(action_space)
        self.share_obs_dim = len(share_obs_space)
        print("obs_space", obs_space)
        print("share_obs_space", share_obs_space)
        print("action_space", action_space)
        user_name = config["args"].user_name
        visualizeClass = importlib.import_module(
            f"envs.{user_name}.{user_name}_visualize"
        )
        self.visualizer = getattr(visualizeClass, "visualizer")

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.all_args.algorithm_name == "MAT":
            next_values = self.trainer.policy.get_values(
                shared_obs=np.concatenate(self.buffer.share_obs[-1]),
                obs=np.concatenate(self.buffer.obs[-1]),
            )
        elif (
            self.all_args.algorithm_name == "RMAPPO"
            or self.all_args.algorithm_name == "IPPO"
        ):
            next_values = self.trainer.policy.get_values(
                shared_obs=np.concatenate(self.buffer.share_obs[-1]),
                rnn_states_critic=np.concatenate(self.buffer.rnn_states_critic[-1]),
                masks=np.concatenate(self.buffer.masks[-1]),
            )
        next_values = np.array(
            np.split(_t2n(next_values), self.all_args.num_rollout_threads)
        )
        self.buffer.compute_returns(
            next_values,
            value_normalizer=self.trainer.value_normalizer,
        )

    def train(
        self,
    ):
        """Train policies with data in the buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos
