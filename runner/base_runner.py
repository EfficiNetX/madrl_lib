import os
import sys

sys.path.append(os.path.abspath("../../"))
from utils.buffer import ReplayBuffer


class BaseRunner(object):
    """MARLアルゴリズムを訓練するためのベースクラス"""

    def __init__(self, config):
        self.args = config["args"]
        self.envs = config["envs"]

        self.share_observation = self.args.share_observation
        self.n_rollout_threads = self.args.n_rollout_threads
        self.num_agents = self.args.num_agents

        # parameters
        self.algorithm_name = self.args.algorithm_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length

        obs_space = self.envs.observation_space[0]  # エージェント0の観測
        share_obs_space = self.envs.share_observation_space[
            0
        ]  # エージェント0のshared観測
        action_space = self.envs.action_space[0]  # 基本的には1次元

        print("obs_space", obs_space)
        print("share_obs_space", share_obs_space)
        print("action_space", action_space)

        # buffer
        self.buffer = ReplayBuffer(
            args=self.args,
            num_agents=self.num_agents,
            obs_space=obs_space,
            share_obs_space=share_obs_space,
            action_space=action_space,
        )
        if self.algorithm_name == "MAT":
            # policy network
            self.policy = None

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError
