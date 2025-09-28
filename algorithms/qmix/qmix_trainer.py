import copy
import torch
import torch.nn as nn
from torch.optim import RMSprop
import numpy as np


class QMIXTrainer:
    def __init__(self, policy, mixer, args, logger=None):
        self.args = args
        self.policy = policy  # QMIXPolicy
        self.mixer = mixer  # QMixer
        self.logger = logger

        self.params = list(policy.agent_q_network.parameters()) + list(
            mixer.parameters()
        )
        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)

        self.optimiser = RMSprop(
            params=self.params,
            lr=args.lr,
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch, t_env, episode_num):
        # batch: EpisodeBatch or dict with keys ["obs", "actions", "reward", "terminated", "state", "avail_actions", "filled"]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Q値計算
        mac_out = []
        self.policy.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.policy.forward(
                batch["obs"][:, t], self.policy.hidden_states
            )
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # [batch, time, agents, actions]

        # 選択した行動のQ値
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)

        # ターゲットQ値計算
        target_mac_out = []
        self.target_policy.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_policy.forward(
                batch["obs"][:, t], self.target_policy.hidden_states
            )
            target_mac_out.append(target_agent_outs)
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)

        # 利用可能な行動以外はマスク
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Double Q-learning
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(
                target_mac_out, 3, cur_max_actions
            ).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mixer
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        # 1-step Q-learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        )

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error**2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip
        )
        self.optimiser.step()

        # ターゲットネットワークの更新
        if (
            episode_num - self.last_target_update_episode
        ) / self.args.qmix_target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # ログ出力
        if (
            self.logger is not None
            and t_env - self.log_stats_t >= self.args.learner_log_interval
        ):
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat(
                "td_error_abs",
                (masked_td_error.abs().sum().item() / mask_elems),
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_policy.agent_q_network.load_state_dict(
            self.policy.agent_q_network.state_dict()
        )
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.policy.agent_q_network.cuda()
        self.target_policy.agent_q_network.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        torch.save(
            self.policy.agent_q_network.state_dict(), f"{path}/agent_q.th"
        )
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        torch.save(self.optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.policy.agent_q_network.load_state_dict(
            torch.load(
                f"{path}/agent_q.th", map_location=lambda storage, loc: storage
            )
        )
        self.target_policy.agent_q_network.load_state_dict(
            torch.load(
                f"{path}/agent_q.th", map_location=lambda storage, loc: storage
            )
        )
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load(
                    f"{path}/mixer.th",
                    map_location=lambda storage, loc: storage,
                )
            )
        self.target_mixer.load_state_dict(
            torch.load(
                f"{path}/mixer.th", map_location=lambda storage, loc: storage
            )
        )
        self.optimiser.load_state_dict(
            torch.load(
                f"{path}/opt.th", map_location=lambda storage, loc: storage
            )
        )
