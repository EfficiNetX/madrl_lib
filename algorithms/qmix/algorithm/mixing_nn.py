import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args, obs_space, share_obs_space, action_space):
        super(QMixer, self).__init__()
        self.args = args
        self.num_agents = args.num_agents
        self.obs_dim = len(obs_space)
        self.shared_obs_dim = len(share_obs_space)
        self.embed_dim = args.qmix_mixer_embed_dim

        # ハイパーネットの層数
        print("obs_dim in QMixer:", self.obs_dim)
        print("num_agents in QMixer:", self.num_agents)
        print("shared_obs_dim in QMixer:", self.shared_obs_dim)
        print("embed_dim in QMixer:", self.embed_dim)
        hypernet_layers = getattr(args, "hypernet_layers", 1)
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.shared_obs_dim, self.embed_dim * self.num_agents)
            self.hyper_w_final = nn.Linear(self.shared_obs_dim, self.embed_dim)
        elif hypernet_layers == 2:
            hypernet_embed = args.qmix_hypernet_embed_dim
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.shared_obs_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.num_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.shared_obs_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise Exception("Error: Only 1 or 2 hypernet layers are supported.")

        self.hyper_b_1 = nn.Linear(self.shared_obs_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.shared_obs_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.to(self.args.device)

    def forward(self, agent_qs, shared_obs):
        """
        agent_qs: (batch_size, n_agents) # 各エージェントのQ値
        shared_obs: (batch_size, shared_obs_dim) # 各エージェントのshared観測
        return: (batch_size, 1) # 全体のQ値
        """
        batch_size = agent_qs.size(0)
        shared_obs = shared_obs.reshape(-1, self.shared_obs_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        

        w1 = th.abs(self.hyper_w_1(shared_obs))
        b1 = self.hyper_b_1(shared_obs)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = th.abs(self.hyper_w_final(shared_obs))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(shared_obs).view(-1, 1, 1)

        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(batch_size, -1, 1)
        return q_tot
