from utils.separated_buffer import SeparatedReplayBuffer


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
            pass
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
