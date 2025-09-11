import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def discrete_autoregreesive_act(
    decoder,
    obs_rep,
    obs,
    batch_size,
    num_agents,
    action_dim,
    tpdv,
    available_actions=None,
    deterministic=False,
):
    shifted_action = torch.zeros(
        (
            batch_size,
            num_agents,
            action_dim + 1,
        )
    ).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, num_agents, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(num_agents):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = (
            distri.probs.argmax(
                dim=-1,
            )
            if deterministic
            else distri.sample()
        )
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < num_agents:
            shifted_action[:, i + 1, 1:] = F.one_hot(
                action,
                num_classes=action_dim,
            )
    return output_action, output_action_log


def discrete_parallel_act(
    decoder,
    obs_rep,
    obs,
    action,
    batch_size,
    num_agents,
    action_dim,
    tpdv,
    available_actions=None,
):
    one_hot_action = F.one_hot(
        action.squeeze(-1),
        num_classes=action_dim,
    )  # (batch, num_agents, action_dim)

    shifted_action = torch.zeros(
        (
            batch_size,
            num_agents,
            action_dim + 1,
        )
    ).to(**tpdv)

    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    # if available_actions is not None:
    #     logit[available_actions == 0] = -1e10 # TODO 不要？

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy
