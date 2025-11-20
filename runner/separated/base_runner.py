import importlib


def _t2n(x):
    return x.detach().cpu().numpy()


class BaseRunner(object):
    """MARLアルゴリズムを訓練するためのベースクラス"""

    def __init__(self, config):
        self.all_args = config["args"]
        self.envs = config["envs"]

        if self.all_args.algorithm_name == "HAPPO":
            from algorithms.happo.happo_trainer import HAPPOTrainer as Trainer
            from algorithms.happo.happpo_pollicy import HAPPO_Policy as Policy
            from utils.separated_buffer import SeparatedReplayBuffer as Buffer
        elif self.all_args.algorithm_name == "HATRPO":
            pass
        elif (
            self.all_args.algorithm_name == "RMAPPO"
            or self.all_args.algorithm_name == "IPPO"
        ):
            from algorithms.r_mappo.algorithm.RMAPPOPolicy import (
                RMAPPOPolicy as Policy,
            )
            from algorithms.r_mappo.rmappo_trainer import RMAPPOTrainer as Trainer
            from utils.separated_buffer import SeparatedReplayBuffer as Buffer
        elif self.all_args.algorithm_name == "HASAC":
            from algorithms.hasac.algorithm.hasac_policy import HASACPolicy as Policy
            from algorithms.hasac.hasac_trainer import HASACTrainer as Trainer
            from utils.offpolicy_separated_buffer import (
                EpisodeReplayBuffer as Buffer,
            )

        self.share_observation_space = (
            self.envs.share_observation_space[0]
            if self.all_args.use_centralized_V
            else self.envs.observation_space[0]
        )

        """policy networkをエージェントの数だけ用意する"""
        self.policy = []
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.all_args.num_agents):
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                self.share_observation_space,
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
            buffer = Buffer(
                args=self.all_args,
                obs_space=self.envs.observation_space[agent_id],
                share_obs_space=self.share_observation_space,
                action_space=self.envs.action_space[agent_id],
            )
            self.buffer.append(buffer)
        # visualizerのimport
        user_name = config["args"].user_name
        visualizeClass = importlib.import_module(
            f"envs.{user_name}.{user_name}_visualize"
        )
        self.visualizer = getattr(visualizeClass, "visualizer")

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
