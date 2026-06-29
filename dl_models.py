from __future__ import annotations

import math
import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_failure = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.node_load = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.graph_risk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        node_logits = self.node_failure(node_embeddings).squeeze(-1)
        load_pred = self.node_load(node_embeddings).squeeze(-1)
        graph_embedding = masked_mean(node_embeddings, mask)
        graph_pred = torch.sigmoid(self.graph_risk(graph_embedding).squeeze(-1))
        return {"node_logits": node_logits, "load_pred": load_pred, "graph_pred": graph_pred}


class MLPBaseline(nn.Module):
    def __init__(self, num_nodes: int, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.num_nodes = num_nodes
        input_dim = num_nodes * feature_dim + num_nodes * num_nodes
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.node_head = nn.Linear(hidden_dim * 2, num_nodes)
        self.load_head = nn.Linear(hidden_dim * 2, num_nodes)
        self.graph_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x, adj, mask = batch["x"], batch["adj"], batch["mask"]
        if x.shape[1] != self.num_nodes or not bool(mask.all()):
            raise ValueError("MLPBaseline requires fixed-size, unpadded graphs.")
        flat = torch.cat([x.flatten(1), adj.flatten(1)], dim=1)
        hidden = self.backbone(flat)
        return {
            "node_logits": self.node_head(hidden),
            "load_pred": self.load_head(hidden),
            "graph_pred": torch.sigmoid(self.graph_head(hidden).squeeze(-1)),
        }


class DenseGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        identity = torch.eye(num_nodes, device=x.device, dtype=adj.dtype).expand(batch_size, -1, -1)
        valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
        a_hat = (adj + identity) * valid_pairs.to(adj.dtype)
        degree = a_hat.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt.unsqueeze(-1) * a_hat * inv_sqrt.unsqueeze(-2)
        output = normalized @ self.linear(x)
        return output * mask.unsqueeze(-1).to(output.dtype)


class GNNPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dimensions = [feature_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            DenseGraphConv(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.heads = PredictionHeads(hidden_dim, dropout)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = batch["x"]
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            updated = torch.relu(layer(hidden, batch["adj"], batch["mask"]))
            if index > 0:
                updated = updated + hidden
            hidden = norm(self.dropout(updated))
            hidden = hidden * batch["mask"].unsqueeze(-1).to(hidden.dtype)
        return hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(batch), batch["mask"])


class StructuralSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_distance: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.distance_bias = nn.Embedding(max_distance + 2, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        distance = dist.clamp(min=0, max=self.max_distance + 1)
        scores = scores + self.distance_bias(distance).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, num_nodes, self.hidden_dim)
        return self.output(output) * mask.unsqueeze(-1).to(output.dtype)


class StructuralTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, max_distance: int, dropout: float):
        super().__init__()
        self.attention = StructuralSelfAttention(hidden_dim, num_heads, max_distance, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, dist, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x * mask.unsqueeze(-1).to(x.dtype)


class HybridGNNTransformer(nn.Module):
    def __init__(
        self, feature_dim: int, pe_dim: int, hidden_dim: int, num_gnn_layers: int,
        transformer_layers: int, transformer_heads: int, ff_dim: int,
        max_distance: int, dropout: float,
    ):
        super().__init__()
        dimensions = [feature_dim] + [hidden_dim] * num_gnn_layers
        self.gnn_layers = nn.ModuleList(
            DenseGraphConv(dimensions[index], dimensions[index + 1])
            for index in range(num_gnn_layers)
        )
        self.gnn_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers))
        self.pe_projection = nn.Linear(pe_dim, hidden_dim)
        self.transformer = nn.ModuleList(
            StructuralTransformerBlock(hidden_dim, transformer_heads, ff_dim, max_distance, dropout)
            for _ in range(transformer_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = PredictionHeads(hidden_dim, dropout)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = batch["x"]
        for index, (layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            updated = torch.relu(layer(hidden, batch["adj"], batch["mask"]))
            if index > 0:
                updated = updated + hidden
            hidden = norm(self.dropout(updated))
        hidden = hidden + self.pe_projection(batch["lap_pe"])
        hidden = hidden * batch["mask"].unsqueeze(-1).to(hidden.dtype)
        for block in self.transformer:
            hidden = block(hidden, batch["dist"], batch["mask"])
        return hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(batch), batch["mask"])


def build_model(model_name: str, metadata: dict, config: dict) -> nn.Module:
    model_cfg = config["model"]
    common = {
        "feature_dim": int(metadata["feature_dim"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "dropout": float(model_cfg["dropout"]),
    }
    if model_name == "mlp":
        return MLPBaseline(int(metadata["max_nodes"]), **common)
    if model_name == "gnn":
        return GNNPredictor(num_layers=int(model_cfg["num_gnn_layers"]), **common)
    if model_name == "hybrid":
        return HybridGNNTransformer(
            pe_dim=int(metadata["lap_pe_dim"]),
            num_gnn_layers=int(model_cfg["num_gnn_layers"]),
            transformer_layers=int(model_cfg["transformer_layers"]),
            transformer_heads=int(model_cfg["transformer_heads"]),
            ff_dim=int(model_cfg["ff_dim"]),
            max_distance=int(config["data"]["max_distance"]),
            **common,
        )
    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
