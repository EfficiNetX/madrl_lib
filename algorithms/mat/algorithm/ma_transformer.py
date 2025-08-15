import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from algorithms.utils.transformer_act import (
    discrete_autoregreesive_act,
    discrete_parallel_act,
)
from algorithms.utils.util import check


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(
        m,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        gain=gain,
    )


class SelfAttention(nn.Module):

    def __init__(self, num_embd, num_head, num_agents, masked=False):
        super(SelfAttention, self).__init__()

        assert num_embd % num_head == 0
        self.masked = masked
        self.num_head = num_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(num_embd, num_embd))
        self.query = init_(nn.Linear(num_embd, num_embd))
        self.value = init_(nn.Linear(num_embd, num_embd))
        # output projection
        self.proj = init_(nn.Linear(num_embd, num_embd))
        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(num_agents + 1, num_agents + 1)).view(
                1, 1, num_agents + 1, num_agents + 1
            ),
        )

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        k = (
            self.key(key)
            .view(
                B,
                L,
                self.num_head,
                D // self.num_head,
            )
            .transpose(1, 2)
        )  # (B, nh, L, hs)
        q = (
            self.query(query)
            .view(
                B,
                L,
                self.num_head,
                D // self.num_head,
            )
            .transpose(1, 2)
        )  # (B, nh, L, hs)
        v = (
            self.value(value)
            .view(
                B,
                L,
                self.num_head,
                D // self.num_head,
            )
            .transpose(1, 2)
        )  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, L, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, num_embd, num_head, num_agents):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)
        self.attn = SelfAttention(num_embd, num_head, num_agents, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(num_embd, 1 * num_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * num_embd, num_embd)),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, num_embd, num_head, num_agents):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)
        self.ln3 = nn.LayerNorm(num_embd)
        self.attn1 = SelfAttention(num_embd, num_head, num_agents, masked=True)
        self.attn2 = SelfAttention(num_embd, num_head, num_agents, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(num_embd, 1 * num_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * num_embd, num_embd)),
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(
        self,
        obs_dim,
        num_block,
        num_embd,
        num_head,
        num_agents,
    ):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.num_embd = num_embd
        self.num_agents = num_agents

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, num_embd), activate=True),
            nn.GELU(),
        )

        self.ln = nn.LayerNorm(num_embd)
        self.blocks = nn.Sequential(
            *[
                EncodeBlock(
                    num_embd,
                    num_head,
                    num_agents,
                )
                for _ in range(num_block)
            ]
        )
        self.head = nn.Sequential(
            init_(nn.Linear(num_embd, num_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(num_embd),
            init_(nn.Linear(num_embd, 1)),
        )

    def forward(self, obs):
        # obs: (batch, num_agents, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_block,
        num_embd,
        num_head,
        num_agents,
        dec_actor=False,
        share_actor=False,
    ):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.num_embd = num_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(
                    nn.LayerNorm(obs_dim),
                    init_(nn.Linear(obs_dim, num_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(num_embd),
                    init_(nn.Linear(num_embd, num_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(num_embd),
                    init_(nn.Linear(num_embd, action_dim)),
                )
            else:
                self.mlp = nn.ModuleList()
                for n in range(num_agents):
                    actor = nn.Sequential(
                        nn.LayerNorm(obs_dim),
                        init_(nn.Linear(obs_dim, num_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(num_embd),
                        init_(nn.Linear(num_embd, num_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(num_embd),
                        init_(nn.Linear(num_embd, action_dim)),
                    )
                    self.mlp.append(actor)
        else:
            self.action_encoder = nn.Sequential(
                init_(
                    nn.Linear(action_dim + 1, num_embd, bias=False),
                    activate=True,
                ),
                nn.GELU(),
            )
            self.obs_encoder = nn.Sequential(
                nn.LayerNorm(obs_dim),
                init_(nn.Linear(obs_dim, num_embd), activate=True),
                nn.GELU(),
            )
            self.ln = nn.LayerNorm(num_embd)
            self.blocks = nn.Sequential(
                *[
                    DecodeBlock(
                        num_embd,
                        num_head,
                        num_agents,
                    )
                    for _ in range(num_block)
                ]
            )
            self.head = nn.Sequential(
                init_(nn.Linear(num_embd, num_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(num_embd),
                init_(nn.Linear(num_embd, action_dim)),
            )

    def forward(self, action, obs_rep, obs):
        # action: (batch, num_agents, action_dim), one-hot/logits?
        # obs_rep: (batch, num_agents, num_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_agents,
        num_block,
        num_embd,
        num_head,
        device=torch.device("cpu"),
        dec_actor=False,
        share_actor=False,
    ):
        super(MultiAgentTransformer, self).__init__()

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        print(obs_dim)

        self.encoder = Encoder(
            obs_dim,
            num_block,
            num_embd,
            num_head,
            num_agents,
        )
        self.decoder = Decoder(
            obs_dim,
            action_dim,
            num_block,
            num_embd,
            num_head,
            num_agents,
            dec_actor=dec_actor,
            share_actor=share_actor,
        )
        self.to(device)

    def forward(self, obs, action, available_actions=None):
        # obs: (batch, num_agents, obs_dim)
        # action: (batch, num_agents, 1)
        # available_actions: (batch, num_agents, act_dim)

        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)
        action = action.long()
        action_log, entropy = discrete_parallel_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            obs=obs,
            action=action,
            batch_size=batch_size,
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            tpdv=self.tpdv,
            available_actions=available_actions,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self,
        obs,
        available_actions=None,
        deterministic=False,
    ):

        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)
        output_action, output_action_log = discrete_autoregreesive_act(
            self.decoder,
            obs_rep,
            obs,
            batch_size,
            self.num_agents,
            self.action_dim,
            self.tpdv,
            available_actions,
            deterministic,
        )
        return output_action, output_action_log, v_loc

    def get_values(self, obs):

        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(obs)
        return v_tot
