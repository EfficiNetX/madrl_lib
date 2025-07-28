import numpy as np
import torch

from runner.base_runner import BaseRunner


class UserEnvRunner(BaseRunner):
    """訓練・評価をするためのクラス"""

    def __init__(self, config):
        super().__init__(config)

    def run(
        self,
    ):
        self.warmup()

        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        for episode in range(episodes):
            # TODO self.use_linear_lr_decay
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

    def warmup(self):
        # envをresetする
        obs = self.envs.reset()
        if self.share_observation:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, axis=1).repeat(
                self.num_agents,
                axis=1,
            )
        else:
            share_obs = obs

        # bufferにshare_obsとobsを格納する
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        pass
