import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from statsmodels.tsa.seasonal import seasonal_decompose
import re
from torch.utils.data import random_split
from sklearn.metrics import mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pkg_resources


logger = logging.getLogger(__name__)



class GATMissingEmbedder(nn.Module):
    """Enhanced GAT for Missing Pattern Embedding"""

    def __init__(self, embedding_size=16, num_heads=4, num_layers=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Dynamically determine the number of layers
        if num_layers is None:
            self.num_layers = min(3, max(1, int(np.log2(embedding_size))))
        else:
            self.num_layers = num_layers

        # Input projection layer
        self.input_proj = nn.Linear(1, embedding_size)

        # Dynamically build GAT layers
        self.gat_layers = nn.ModuleList()
        in_dim = embedding_size

        for i in range(self.num_layers):
            out_dim = embedding_size if i == self.num_layers - 1 else embedding_size // 2
            heads = num_heads if i < self.num_layers - 1 else 1  # Single head for the last layer

            self.gat_layers.append(GATConv(
                in_dim, out_dim, heads=heads,
                dropout=0.2, concat=(i < self.num_layers - 1)
            ))
            in_dim = out_dim * heads

        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=2,
            dropout=0.1
        )

    def forward(self, x, edge_index, context_mask=None):
        # Initial projection
        x = F.elu(self.input_proj(x))

        # GAT layers
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)

        # Context attention
        if context_mask is not None:
            x = x.unsqueeze(0)  # Add batch dimension
            attn_out, _ = self.context_attention(
                x, x, x,
                key_padding_mask=~context_mask
            )
            x = attn_out.squeeze(0)

        return x

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()


class MissingGAT(nn.Module):
    """Graph Attention Network for Missing Pattern Analysis"""

    def __init__(self, in_features, out_features=8, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        self.out_proj = nn.Linear(out_features * num_heads, out_features)

    def forward(self, x, edge_index):
        batch_size = x.size(0)

        q = self.query(x).view(batch_size, -1, self.num_heads, self.out_features)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.out_features)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.out_features)

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.out_features ** 0.5)

        adj_matrix = torch.zeros((batch_size, batch_size), device=x.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        attn_scores = attn_scores.masked_fill(adj_matrix == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(batch_size, -1, self.num_heads * self.out_features)
        return self.out_proj(out)