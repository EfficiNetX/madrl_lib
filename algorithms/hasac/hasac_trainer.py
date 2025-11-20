class BaseSACTrainer:
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy

    def train(self, batch):
        """共通のtrainフロー"""
        self.train_critic(batch)
        self.train_actor(batch)
        self.update_target_networks()

    def train_critic(self, batch):
        raise NotImplementedError

    def train_actor(self, batch):
        raise NotImplementedError


class CentralizedSACTrainer(BaseSACTrainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)
        # 集中型Critic（全エージェント共通）


# 分散型（Independent SAC）
class IndependentSACTrainer(BaseSACTrainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)
        # 各エージェントごとにCriticを持つ
