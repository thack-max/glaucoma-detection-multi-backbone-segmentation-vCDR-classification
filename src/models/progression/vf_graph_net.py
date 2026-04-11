"""
Visual Field Graph Isomorphism Network (VF-GIN) for glaucoma defect detection.

Reference: "Convolutional Graph Isomorphism Network to Detect Glaucomatous
Visual Field Defects." Ophthalmology Science, 2025.
  - 1,874 tests, 1,009 eyes, 676 patients
  - AUC 0.982, sensitivity 94.1% at 95% specificity
  - Significantly outperforms Anderson criteria (0.906), GHT/PSD (0.936),
    dense NN (0.941), and CNN (0.941) — all p < 0.05

Why a graph network for VF data:
  - The 24-2 Humphrey Visual Field test measures sensitivity at 54 points
    in a specific spatial arrangement on the retina.
  - Adjacent test points share nerve fiber bundles — damage propagates
    along these anatomical pathways.
  - A CNN treats VF as a regular grid, ignoring the irregular spacing and
    the underlying nerve fiber topology.
  - A GIN models each test point as a node with edges encoding spatial
    adjacency, naturally representing the VF's anatomical structure.
  - GraphNOSE explainability highlights which VF regions drive predictions,
    revealing clinically meaningful arcuate patterns.

Architecture:
  - 54 nodes = 54 VF test points
  - Node features: sensitivity (dB), total deviation, pattern deviation
  - Edges: spatial adjacency based on 24-2 test point layout
  - Graph Isomorphism Network (GIN) layers with MLP aggregation
  - Global readout via mean + max pooling
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# 24-2 Humphrey VF test point coordinates (approximate grid positions)
# These define the spatial layout for constructing adjacency edges.
# Format: (row, col) on the standard 24-2 grid.
HVF_24_2_COORDS = [
    # Row 0 (superior)
    (0, 3), (0, 4), (0, 5), (0, 6),
    # Row 1
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    # Row 2
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    # Row 3
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8),
    # Row 4 (blind spot gap at (3,4) and (4,4))
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8),
    # Row 5
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
    # Row 6
    (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
    # Row 7 (inferior)
    (7, 3), (7, 4), (7, 5), (7, 6),
]

NUM_VF_POINTS = 54


def build_adjacency_edges(
    coords: list[tuple[int, int]] | None = None,
    max_distance: float = 1.5,
) -> torch.Tensor:
    """Build edge index tensor from VF test point coordinates.

    Two points are connected if their Euclidean distance on the grid
    is <= max_distance (default 1.5 connects direct neighbors including
    diagonals on the 24-2 grid).

    Returns:
        Edge index tensor [2, num_edges] in COO format.
    """
    if coords is None:
        coords = HVF_24_2_COORDS

    n = len(coords)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dr = coords[i][0] - coords[j][0]
            dc = coords[i][1] - coords[j][1]
            dist = math.sqrt(dr * dr + dc * dc)
            if dist <= max_distance:
                edges.append((i, j))
                edges.append((j, i))  # Undirected

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer.

    GIN update rule: h_v^{k+1} = MLP((1 + eps) * h_v^k + sum_{u in N(v)} h_u^k)

    GIN is provably as powerful as the Weisfeiler-Leman graph isomorphism test,
    making it strictly more expressive than GCN or GraphSAGE for graph
    classification tasks.
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0, train_eps: bool = True) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps)) if train_eps else eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """GIN message passing.

        Args:
            x: Node features [num_nodes, in_dim].
            edge_index: Edge indices [2, num_edges] in COO format.

        Returns:
            Updated node features [num_nodes, out_dim].
        """
        N = x.shape[0]

        # Aggregate neighbor features
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(N, x.shape[1], device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])

        # GIN update
        out = (1 + self.eps) * x + agg
        return self.mlp(out)


class VFGraphNet(nn.Module):
    """Visual Field Graph Isomorphism Network for glaucoma defect detection.

    Processes 24-2 Humphrey VF test data as a graph, preserving the
    spatial topology of test points and their nerve fiber bundle relationships.

    AUC 0.982 for detecting glaucomatous VF defects (reference implementation).
    """

    def __init__(
        self,
        node_features: int = 3,  # sensitivity, total deviation, pattern deviation
        hidden_dim: int = 128,
        num_gin_layers: int = 4,
        output_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,  # normal vs glaucomatous
    ) -> None:
        """
        Args:
            node_features: Number of features per VF test point.
            hidden_dim: GIN hidden layer dimension.
            num_gin_layers: Number of GIN convolution layers.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
            num_classes: Classification classes.
        """
        super().__init__()
        self.output_dim = output_dim

        # Build adjacency structure
        self.register_buffer("edge_index", build_adjacency_edges())

        # Node feature embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(num_gin_layers):
            in_d = hidden_dim
            out_d = hidden_dim if i < num_gin_layers - 1 else output_dim
            self.gin_layers.append(GINConv(in_d, out_d))

        self.dropout = nn.Dropout(dropout)

        # Graph-level readout (mean + max pooling concatenated)
        self.readout_dim = output_dim * 2

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_graph(self, vf_data: torch.Tensor) -> torch.Tensor:
        """Encode VF test points as a graph and produce a graph-level embedding.

        Args:
            vf_data: [B, num_points, node_features] — VF measurements.
                     num_points should be 54 for 24-2 HVF.

        Returns:
            Graph-level embedding [B, readout_dim].
        """
        B, N, F = vf_data.shape

        # Process each sample's graph
        outputs = []
        for b in range(B):
            x = self.node_embed(vf_data[b])  # [N, hidden_dim]

            for gin_layer in self.gin_layers:
                x = gin_layer(x, self.edge_index)
                x = self.dropout(x)

            # Graph-level readout: mean + max pooling
            mean_pool = x.mean(dim=0)
            max_pool = x.max(dim=0).values
            graph_embed = torch.cat([mean_pool, max_pool], dim=-1)
            outputs.append(graph_embed)

        return torch.stack(outputs)

    def forward(self, vf_data: torch.Tensor) -> torch.Tensor:
        """Classify VF data as normal or glaucomatous.

        Args:
            vf_data: [B, 54, 3] — VF measurements per test point.

        Returns:
            Classification logits [B, num_classes].
        """
        graph_embed = self.encode_graph(vf_data)
        return self.classifier(graph_embed)

    def get_node_importance(self, vf_data: torch.Tensor) -> torch.Tensor:
        """Compute per-node importance scores for explainability.

        Provides a GraphNOSE-like visualization showing which VF regions
        contribute most to the glaucoma prediction — should highlight
        arcuate regions in clinically accurate patterns.

        Args:
            vf_data: [B, 54, 3].

        Returns:
            Node importance scores [B, 54].
        """
        B, N, F = vf_data.shape
        importance = []

        for b in range(B):
            x = self.node_embed(vf_data[b])
            for gin_layer in self.gin_layers:
                x = gin_layer(x, self.edge_index)
            # L2 norm per node as importance proxy
            node_scores = x.norm(dim=-1)
            importance.append(node_scores)

        return torch.stack(importance)
