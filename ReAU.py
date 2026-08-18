import torch
from torch import nn
import utility.trainer as trainer
import utility.tools as tools
import utility.losses as losses
import torch.nn.functional as F
import math
import random


class ReAU(nn.Module):
    def __init__(self, args, dataset, device):
        super(ReAU, self).__init__()
        self.model_name = "ReAU"
        self.dataset = dataset
        self.args = args
        self.device = device
        self.reg_lambda = float(self.args.reg_lambda)
        self.activation = nn.Sigmoid()
        self.ssl_lambda = float(self.args.ssl_lambda)
        self.tau = float(self.args.tau)
        self.align_z = int(getattr(self.args, 'align_z', 1))
        self.encoder = self.args.encoder
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.args.embedding_size))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.args.embedding_size))
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
        self.activation_layer = nn.Tanh()

        self.adj_mat = self.dataset.sparse_adjacency_matrix()
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = []
        if self.args.dataset == 'amazon-book':
            all_embeddings = [embeddings]
        for _ in range(int(self.args.gcn_layer)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return user_emb, item_emb

    def alignment(self, embedding1, embedding2, alpha=2):
        embedding1 = torch.nn.functional.normalize(embedding1, dim=-1)
        embedding2 = torch.nn.functional.normalize(embedding2, dim=-1)

        return torch.mean(((embedding1 - embedding2)).norm(p=2, dim=1).pow(alpha))

    def uniformity(self, embedding, t=2):
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return torch.pdist(embedding, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, local_space):
        if self.encoder == 'MF':
            all_user_embed, all_item_embed = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_embed, all_item_embed = self.aggregate()

        awn = local_space.awn
        anchor = awn[:, 0]
        neighbor = awn[:, 1]

        anchor_embed = all_user_embed[anchor.long()]
        neighbor_embed = all_item_embed[neighbor.long()]

        align_loss = losses.multi_local_align(anchor_embed, neighbor_embed, tau=self.tau, z=self.align_z)

        user_space_embed = all_user_embed[local_space.uniform_user_space.long()]
        item_space_embed = all_item_embed[local_space.uniform_item_space.long()]

        uniform_loss = (losses.rebalanced_local_uniform(user_space_embed) + losses.rebalanced_local_uniform(
            item_space_embed)) / 2

        loss_list = [align_loss, self.ssl_lambda * uniform_loss]

        return loss_list

    def get_rating_for_test(self, user):
        if self.encoder == 'MF':
            all_user_embed, all_item_embed = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_embed, all_item_embed = self.aggregate()

        user_embed = all_user_embed[user.long()]

        rating = self.activation(torch.matmul(user_embed, all_item_embed.t()))

        return rating


class Trainer():
    def __init__(self, args, dataset, device, logger):
        self.model = ReAU(args, dataset, device)
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger

    def train(self):
        trainer.training(self.model, self.args, self.dataset, self.device, self.logger)
