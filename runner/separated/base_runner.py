from utils.separated_buffer import SeparatedReplayBuffer
import torch
import numpy as np


def _t2n(x):
    return x.detach().cpu().numpy()


class BaseRunner(object):
    """MARLアルゴリズムを訓練するためのベースクラス"""

    def __init__(self, config):
        self.all_args = config["args"]
        self.envs = config["envs"]

        self.share_observation = self.all_args.share_observation
        self.num_rollout_threads = self.all_args.num_rollout_threads
        self.num_agents = self.all_args.num_agents
        self.use_centralized_V = self.all_args.use_centralized_V
        self.hidden_size = self.all_args.hidden_size
        self.recurrent_N = self.all_args.recurrent_N
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # parameters
        self.algorithm_name = self.all_args.algorithm_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length

        # interval
        self.log_interval = self.all_args.log_interval

        if self.algorithm_name == "HAPPO":
            from algorithms.happo.happo_trainer import HAPPOTrainer as Trainer
            from algorithms.happo.happpo_pollicy import HAPPO_Policy as Policy
        elif self.algorithm_name == "HATRPO":
            pass
        else:
            from algorithms.r_mappo.rmappo_trainer import RMAPPOTrainer as Trainer
            from algorithms.r_mappo.algorithm.RMAPPOPolicy import (
                RMAPPOPolicy as Policy,
            )

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        """policy networkをエージェントの数だけ用意する"""
        self.policy = []
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.policy.append(po)
            # algorithm
            trainer = Trainer(
                args=self.all_args,
                policy=self.policy[agent_id],
            )
            self.trainer.append(trainer)
            # buffer
            buffer = SeparatedReplayBuffer(
                args=self.all_args,
                obs_space=self.envs.observation_space[agent_id],
                share_obs_space=share_observation_space,
                action_space=self.envs.action_space[agent_id],
            )
            self.buffer.append(buffer)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_values = self.trainer[agent_id].policy.get_values(
                shared_obs=self.buffer[agent_id].share_obs[-1],
                rnn_states_critic=self.buffer[agent_id].rnn_states_critic[-1],
                masks=self.buffer[agent_id].masks[-1],
            )
            next_values = _t2n(next_values)
            self.buffer[agent_id].compute_returns(
                next_values, self.trainer[agent_id].value_normalizer
            )

    def train(self):
        train_infos = []
        factor = np.ones(
            (self.episode_length, self.num_rollout_threads, 1), dtype=np.float32
        )
        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor) 
            if self.all_args.algorithm_name == "HATRPO":
                pass
            else:
                old_actions_log_prob, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs=self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    rnn_states=self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    action=self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    masks=self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                )
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            if self.all_args.algorithm_name == "HATRPO":
                pass
            else:
                new_actions_log_prob, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs=self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    rnn_states=self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    action=self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    masks=self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                )
            factor = factor * _t2n(
                torch.prod(
                    torch.exp(new_actions_log_prob - old_actions_log_prob), dim=-1
                ).reshape(self.episode_length, self.num_rollout_threads, 1)
            )
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos
