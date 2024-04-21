from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, ARGA
from torch_geometric.nn.conv import GCNConv, APPNP, SAGEConv, GATConv, GATv2Conv
from torch_geometric.utils import negative_sampling


class GAE_NA(GAE):
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   self_loop_index: Optional[Tensor] = None,
                   num_per_loop: Optional[Tensor] = None,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        EPS = 1e-15
        num_samples = pos_edge_index.shape[1]
        if self_loop_index is None:
            pos_loss = -torch.log(
                self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        else:
            num_samples += num_per_loop.sum()
            pos_loss = - 1 / num_samples * (
                    (self.decoder(z, pos_edge_index, sigmoid=True) + EPS).log().sum() + (
                    (self.decoder(z, self_loop_index, sigmoid=True) + EPS).log() * num_per_loop).sum())

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_samples)
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


class ARGA_NA(ARGA):
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   self_loop_index: Optional[Tensor] = None,
                   num_per_loop: Optional[Tensor] = None,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        EPS = 1e-15
        num_samples = pos_edge_index.shape[1]
        if self_loop_index is None:
            pos_loss = -torch.log(
                self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        else:
            num_samples += num_per_loop.sum()
            pos_loss = - 1 / num_samples * (
                    (self.decoder(z, pos_edge_index, sigmoid=True) + EPS).log().sum() + (
                    (self.decoder(z, self_loop_index, sigmoid=True) + EPS).log() * num_per_loop).sum())

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_samples)
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, cached=True)
        self.activation = nn.ReLU()
        self.conv2 = GCNConv(hid_dim, out_dim, cached=True)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.activation(z)
        z = self.conv2(z, edge_index)
        return z


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.activation = nn.ReLU()
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.activation(z)
        z = self.conv2(z, edge_index)
        return z


class GATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_dim, out_dim, heads=8, concat=True)
        self.activation = nn.ReLU()
        self.conv2 = GATConv(out_dim * 8, out_dim, concat=False)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.activation(z)
        z = self.conv2(z, edge_index)
        return z


class GATv2Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_dim, out_dim, heads=8, concat=True)
        self.activation = nn.ReLU()
        self.conv2 = GATv2Conv(out_dim * 8, out_dim, concat=False)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.activation(z)
        z = self.conv2(z, edge_index)
        return z


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLPEncoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        z = self.linear1(x)
        z = self.activation(z)
        z = self.linear2(z)
        return z


class Discriminator(nn.Module):
    """
    [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407)
    from Pan *et al.* (IJCAI 2018)
    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class LinearGCNEncoder(nn.Module):
    """
    [Keep It Simple: Graph Autoencoders Without Graph Convolutional Networks](https://arxiv.org/abs/1910.00942)
    from Salha *et al.* (NeurIPS-W 2019, then ECML-PKDD 2020 - [see here](https://arxiv.org/abs/2001.07614))
    """

    def __init__(self, in_dim, out_dim):
        super(LinearGCNEncoder, self).__init__()
        self.conv = GCNConv(in_dim, out_dim, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GNCNEncoder(nn.Module):
    """
    [Variational Graph Normalized AutoEncoders](https://arxiv.org/abs/2108.08046)
    from Seong Jin Ahn and MyoungHo Kim (CIKM 2021)
    """

    def __init__(self, in_dim, out_dim, scaling_factor: float = 1.8):
        super(GNCNEncoder, self).__init__()
        self.scaling_factor = scaling_factor
        self.linear = nn.Linear(in_dim, out_dim)
        self.propagate = APPNP(K=1, alpha=0, cached=True)

    def forward(self, x, edge_index):
        z = self.linear(x)
        z = F.normalize(z, p=2, dim=1) * self.scaling_factor
        z = self.propagate(z, edge_index)
        return z
