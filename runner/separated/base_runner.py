import importlib


def _t2n(x):
    return x.detach().cpu().numpy()


class BaseRunner(object):
    """MARLアルゴリズムを訓練するためのベースクラス"""

    def __init__(self, config):
        self.all_args = config["args"]
        self.envs = config["envs"]
        self.share_observation_space = (
            self.envs.share_observation_space[0]
            if self.all_args.use_centralized_V
            else self.envs.observation_space[0]
        )
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
