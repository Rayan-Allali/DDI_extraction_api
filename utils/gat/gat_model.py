import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import math
from torch_geometric.nn import global_add_pool, global_mean_pool


VIEW1_INPUT_SIZE =1068
class MultiViewCrossAttention(nn.Module):
    def __init__(self, embed_dim, cls_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(cls_dim if cls_dim else embed_dim, embed_dim)
        self.v_linear = nn.Linear(cls_dim if cls_dim else embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        assert Q.size(-1) == self.embed_dim, f"Expected Q dimension {self.embed_dim}, got {Q.size(-1)}"
        if K is not None:
            assert K.size(-1) == (self.k_linear.in_features), f"Expected K dimension {self.k_linear.in_features}, got {K.size(-1)}"
        if V is not None:
            assert V.size(-1) == (self.v_linear.in_features), f"Expected V dimension {self.v_linear.in_features}, got {V.size(-1)}"

        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        context = self.out_linear(context)  # Apply output projection bofere normalisation try this first then try to place this after norm layer to see wich approach is better
        context = self.layer_norm(context)
        return context

class GATModelWithAttention(nn.Module):
    def __init__(self, node_in_dim, gat_hidden_channels, cls_dim, dropout_rate,num_classes=5):
        super().__init__()

        self.gat1 = GATv2Conv(node_in_dim, gat_hidden_channels, heads=4, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(gat_hidden_channels * 4)
        self.gat2 = GATv2Conv(gat_hidden_channels * 4, gat_hidden_channels, heads=4, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(gat_hidden_channels * 4)
        self.gat3 = GATv2Conv(gat_hidden_channels * 4, gat_hidden_channels, heads=4, dropout=dropout_rate)
        self.bn3 = nn.BatchNorm1d(gat_hidden_channels * 4)
        self.cross_attention = MultiViewCrossAttention(gat_hidden_channels * 4, cls_dim)

        self.ffn = nn.Sequential(
           nn.Linear(gat_hidden_channels * 4, gat_hidden_channels * 4),
           nn.ReLU(),
           nn.Dropout(p=dropout_rate),
          nn.Linear(gat_hidden_channels * 4, gat_hidden_channels * 4)
        )
        self.fc_out = nn.Linear(gat_hidden_channels * 4, num_classes)
        self.ffn_norm = nn.LayerNorm(gat_hidden_channels * 4)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x)
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x)
        node_features = []
        add_pool = global_add_pool(x, batch)
        mean_pool = global_mean_pool(x, batch)

        node_features = (add_pool + mean_pool) / 2
        biobert_cls = data.biobert_cls.view(-1, 768)
        attn_output = self.cross_attention(node_features, biobert_cls, biobert_cls)
        # attn_output = F.dropout(attn_output, p=0.2)
        ffn_output = self.ffn(attn_output)
        attn_output = attn_output + ffn_output
        attn_output = self.ffn_norm(attn_output)
        logits = self.fc_out(attn_output).squeeze(1)

        return logits