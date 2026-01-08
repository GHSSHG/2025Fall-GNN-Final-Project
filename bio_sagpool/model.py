from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

from .dataset import AA_VOCAB


class SagPoolBindingNet(nn.Module):
    def __init__(
        self,
        *,
        aa_vocab_size: int = len(AA_VOCAB),
        aa_emb_dim: int = 32,
        hidden_dim: int = 128,
        pooling_ratio: float = 0.5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(aa_vocab_size, aa_emb_dim)
        in_dim = aa_emb_dim + 2  # pos + is_cdr3

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=pooling_ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPooling(hidden_dim, ratio=pooling_ratio)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPooling(hidden_dim, ratio=pooling_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        aa = data.aa
        x = self.embedding(aa)
        x = torch.cat([x, data.pos, data.is_cdr3], dim=1)

        edge_index = data.edge_index
        batch = data.batch

        x = torch.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        g = x1 + x2 + x3
        logits = self.mlp(g).view(-1)
        return logits

