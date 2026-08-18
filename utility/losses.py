import torch
import torch.nn.functional as F
import numpy as np


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss


def multi_local_align(embedding1, embedding2, alpha=2, tau=0.25, z=4, max_power=4):
    if z < 1 or z > max_power:
        raise ValueError(f"z must be in [1, {max_power}], got {z}")

    x = F.normalize(embedding1, dim=-1)
    y = F.normalize(embedding2, dim=-1)
    cos = (x * y).sum(dim=-1)

    terms = torch.stack([cos.pow(power) / tau for power in range(1, max_power + 1)], dim=-1)
    mask = torch.zeros(max_power, dtype=torch.bool, device=cos.device)
    mask[0] = True
    mask[z - 1] = True

    cos = torch.logsumexp(terms.masked_fill(~mask, float('-inf')), dim=-1)
    return -cos.mean()


def multi_local_align_1(embedding1, embedding2, alpha=2, tau=0.25):
    return multi_local_align(embedding1, embedding2, alpha=alpha, tau=tau, z=1)


def multi_local_align_2(embedding1, embedding2, alpha=2, tau=0.25):
    return multi_local_align(embedding1, embedding2, alpha=alpha, tau=tau, z=2)


def multi_local_align_3(embedding1, embedding2, alpha=2, tau=0.25):
    return multi_local_align(embedding1, embedding2, alpha=alpha, tau=tau, z=3)


def multi_local_align_4(embedding1, embedding2, alpha=2, tau=0.25):
    return multi_local_align(embedding1, embedding2, alpha=alpha, tau=tau, z=4)


def rebalanced_local_uniform(embedding, tau=0.25, eps=1e-12):
    x = torch.nn.functional.normalize(embedding, dim=-1)  # ensure norm=1
    n = x.size(0)
    sim = x @ x.t()
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
    sim_pairs = sim[mask]
    vals = sim_pairs / tau
    maxv = vals.max()
    logsum = (vals - maxv).exp().sum().log() + maxv
    logmean = logsum - torch.log(torch.tensor(sim_pairs.numel(), device=x.device, dtype=logsum.dtype))
    return logmean



