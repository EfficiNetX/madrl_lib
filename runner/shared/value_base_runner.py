from runner.shared.base_runner import BaseRunner


class ValueBaseRunner(BaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDNなど）用のベースクラス
    """

    def __init__(self, config):
        super().__init__(config)
        # 追加で必要な初期化（state_dim, mixer, etc.）
        self.state_dim = self.envs.share_observation_space[0]
        # ここでValue系専用のpolicy, mixer, trainer, bufferを初期化
        if self.algorithm_name == "QMIX":
            from algorithms.qmix.algorithm.qmix_policy import (
                QMIXPolicy as Policy,
            )
            from algorithms.qmix.algorithm.mixing_nn import QMixer as Mixer
            from algorithms.qmix.qmix_trainer import QMIXTrainer as Trainer
            from utils.shared_episode_buffer import (
                EpisodeReplayBuffer as Buffer,
            )
        else:
            raise NotImplementedError("Unknown value-based algorithm.")
        # TODO: 中央集権型価値関数を使わない場合の処理．self.mixer = Noneの時のtrainerの挙動を実装すれば良い
        self.policy = Policy(self.all_args, ...)
        self.mixer = Mixer(self.all_args)
        self.trainer = Trainer(self.policy, self.mixer, self.all_args)
        self.buffer = Buffer(self.all_args, ...)

    def collect(self, step):
        # 1ステップ分のデータ収集
        actions, next_hidden_states = self.policy.select_actions(
            self.current_obs, self.current_hidden_states
        )
        return actions, next_hidden_states

    def warmup(self):
        # 環境をリセットして初期観測を取得
        obs = self.envs.reset()
        # 必要なら初期観測をバッファや変数に保存
        self.initial_obs = obs.copy()
        # hidden stateの初期化も必要なら
        self.initial_hidden_states = self.policy.init_hidden(
            batch_size=self.num_rollout_threads
        )

    def insert(self, episode_data):
        self.buffer.add(episode_data)

    def train(self, t_env, episode_num):
        if self.buffer.can_sample(self.all_args.qmix_batch_size):
            batch = self.buffer.sample(self.all_args.qmix_batch_size)
            self.trainer.train(batch, t_env=t_env, episode_num=episode_num)

    def update_epsilon(self, t_env):
        self.policy.update_epsilon(t_env)

    def can_sample(self):
        return self.buffer.can_sample(self.all_args.qmix_batch_size)
