import torch
import torch.nn as nn
from ma_transformer import MultiAgentTransformer


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        args,
        obs_space,
        action_space,
    ):
        super(TransformerPolicy, self).__init__()
        self.args = args
        self.obs_space = obs_space
        self.action_space = action_space

        self.obs_dim = obs_space[0]
        self.action_dim = action_space[0]

        print("obs_dim", self.obs_dim)
        print("action_dim", self.action_dim)
        if self.action_type == "Discrete":
            self.act_dim = len(action_space)
            self.act_num = 1

        self.num_agents = self.args.num_agents

        self.tpdv = dict(dtype=torch.float32, device=self.args.device)

        self.transformer = MultiAgentTransformer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            num_block=self.args.num_block,
            num_embd=self.args.num_embd,
            num_head=self.args.num_head,
            device=self.args.device,
            action_type="Discrete",
            dec_actor=False,
            share_actor=False,
        )

        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
