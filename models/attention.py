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


class CrossTaskAttention(nn.Module):
    def __init__(self, hidden_size, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.query_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_tasks)])
        self.key_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_tasks)])
        self.value_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_tasks)])

    def forward(self, task_features):
        all_attended = []
        for i in range(self.num_tasks):
            queries = self.query_proj[i](task_features[i].unsqueeze(0))
            keys = torch.stack([self.key_proj[i](feat) for feat in task_features])
            values = torch.stack([self.value_proj[i](feat) for feat in task_features])
            attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
            attn_weights = F.softmax(attn_scores / (self.hidden_size ** 0.5), dim=-1)
            attended = torch.matmul(attn_weights, values).squeeze(0)
            all_attended.append(attended)
        return torch.stack(all_attended)
