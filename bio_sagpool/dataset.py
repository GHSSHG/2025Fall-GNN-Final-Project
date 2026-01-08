from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

AMINO_ACIDS_20 = "ACDEFGHIKLMNPQRSTVWY"
UNK_AA = "X"
AA_VOCAB = list(AMINO_ACIDS_20) + [UNK_AA]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


def _clean_seq(seq: str) -> str:
    return str(seq).strip().upper()


def _seq_to_idx(seq: str) -> torch.Tensor:
    seq = _clean_seq(seq)
    indices = [AA_TO_IDX.get(ch, AA_TO_IDX[UNK_AA]) for ch in seq]
    return torch.tensor(indices, dtype=torch.long)


@dataclass(frozen=True)
class _GraphTemplate:
    edge_index: torch.Tensor  # [2, E] (CPU)
    pos: torch.Tensor  # [N, 1] (CPU)
    is_cdr3: torch.Tensor  # [N, 1] (CPU)


class PairSequenceGraphCache:
    """Caches graph templates by (len_epitope, len_cdr3) to speed up dataset access."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], _GraphTemplate] = {}

    def get(self, len_epitope: int, len_cdr3: int) -> _GraphTemplate:
        key = (int(len_epitope), int(len_cdr3))
        tpl = self._cache.get(key)
        if tpl is not None:
            return tpl

        if len_epitope <= 0 or len_cdr3 <= 0:
            raise ValueError(f"Empty sequence length: len_epitope={len_epitope}, len_cdr3={len_cdr3}")

        num_nodes = len_epitope + len_cdr3
        offset = len_epitope

        edge_parts = []

        # Intra-sequence edges (undirected chain)
        if len_epitope >= 2:
            src = torch.arange(len_epitope - 1, dtype=torch.long)
            dst = src + 1
            edge_parts.append(torch.stack([src, dst], dim=0))
            edge_parts.append(torch.stack([dst, src], dim=0))
        if len_cdr3 >= 2:
            src = torch.arange(len_cdr3 - 1, dtype=torch.long) + offset
            dst = src + 1
            edge_parts.append(torch.stack([src, dst], dim=0))
            edge_parts.append(torch.stack([dst, src], dim=0))

        # Cross edges (complete bipartite, undirected)
        epi_nodes = torch.arange(len_epitope, dtype=torch.long)
        cdr_nodes = torch.arange(len_cdr3, dtype=torch.long) + offset
        cross_src = epi_nodes.repeat_interleave(len_cdr3)
        cross_dst = cdr_nodes.repeat(len_epitope)
        edge_parts.append(torch.stack([cross_src, cross_dst], dim=0))
        edge_parts.append(torch.stack([cross_dst, cross_src], dim=0))

        edge_index = torch.cat(edge_parts, dim=1).contiguous()

        # Node-level static features.
        def _pos_norm(length: int) -> torch.Tensor:
            if length <= 1:
                return torch.zeros(length, dtype=torch.float32)
            return torch.arange(length, dtype=torch.float32) / float(length - 1)

        pos = torch.cat([_pos_norm(len_epitope), _pos_norm(len_cdr3)], dim=0).view(num_nodes, 1)
        is_cdr3 = torch.zeros(num_nodes, 1, dtype=torch.float32)
        is_cdr3[offset:, 0] = 1.0

        tpl = _GraphTemplate(edge_index=edge_index, pos=pos, is_cdr3=is_cdr3)
        self._cache[key] = tpl
        return tpl


class EpitopeCdr3BindingDataset(Dataset):
    """Dataset of (Epitope, CDR3) pairs as a single interaction graph."""

    def __init__(
        self,
        epitopes: Sequence[str],
        cdr3s: Sequence[str],
        labels: Sequence[int | float],
        graph_cache: PairSequenceGraphCache | None = None,
    ) -> None:
        if len(epitopes) != len(cdr3s) or len(epitopes) != len(labels):
            raise ValueError("epitopes, cdr3s, labels must have the same length")
        self.epitopes = list(epitopes)
        self.cdr3s = list(cdr3s)
        self.labels = list(labels)
        self.graph_cache = graph_cache or PairSequenceGraphCache()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Data:
        ep = _clean_seq(self.epitopes[idx])
        cdr = _clean_seq(self.cdr3s[idx])
        aa = torch.cat([_seq_to_idx(ep), _seq_to_idx(cdr)], dim=0)

        tpl = self.graph_cache.get(len(ep), len(cdr))

        # Important: clone cached tensors to avoid in-place device moves from sharing.
        data = Data(
            aa=aa,
            pos=tpl.pos.clone(),
            is_cdr3=tpl.is_cdr3.clone(),
            edge_index=tpl.edge_index.clone(),
            y=torch.tensor([float(self.labels[idx])], dtype=torch.float32),
        )
        data.num_nodes = aa.numel()
        return data

