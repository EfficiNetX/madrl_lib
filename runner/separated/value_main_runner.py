import time
from runner.separated.value_base_runner import ValueBaseRunner


class ValueMainRunner(ValueBaseRunner):
    """
    Value-basedアルゴリズム（QMIX/VDN等）用のメインRunnerクラス
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.num_rollout_threads
        )

        for episode in range(episodes):
            episode_data = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "state": [],
                "avail_actions": [],
                "filled": [],
            }
            obs = self.envs.reset()
            hidden_states = self.policy.init_hidden(batch_size=1)
            for step in range(self.episode_length):
                actions, next_hidden_states = self.policy.select_actions(
                    obs, hidden_states
                )
                next_obs, rewards, dones, info = self.envs.step(actions)
                episode_data["obs"].append(obs)
                episode_data["actions"].append(actions)
                episode_data["rewards"].append(rewards)
                episode_data["terminated"].append(dones)
                # state, avail_actions, filledも同様に保存
                obs = next_obs
                hidden_states = next_hidden_states
                if dones.any():
                    break
            self.insert(episode_data)
            self.train(t_env=episode, episode_num=episode)
            self.update_epsilon(episode)

            if episode % self.log_interval == 0:
                print(f"Episode {episode}/{episodes}")
