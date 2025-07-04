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

class MissingClusterAnalyzer:
    """Missing Subgroup Clustering Analyzer"""
    def __init__(self, n_clusters=5):
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_clusters)
        )
        self.cluster = DBSCAN(eps=0.5, min_samples=5)

    def fit(self, X):
        """Fit the clustering model"""
        try:
            # First, reduce dimensionality with PCA
            pca = PCA(n_components=64)
            X_pca = pca.fit_transform(X.fillna(0))

            # Further dimensionality reduction with autoencoder
            if isinstance(X_pca, np.ndarray):
                X_pca = torch.tensor(X_pca, dtype=torch.float32)
            embeddings = self.encoder(X_pca)
            self.cluster.fit(embeddings.detach().numpy())
            logger.info(f"Clustering complete, found {len(np.unique(self.cluster.labels_))} clusters")
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            raise

    def analyze_clusters(self, df):
        """Analyze clustering results"""
        try:
            if not hasattr(self.cluster, 'labels_'):
                raise ValueError("Please call the fit method first to perform clustering")

            df = df.copy()
            df['missing_cluster'] = self.cluster.labels_
            cluster_stats = []

            for cluster_id in np.unique(self.cluster.labels_):
                if cluster_id == -1:
                    continue

                cluster_data = df[df['missing_cluster'] == cluster_id]
                stats = {
                    'cluster': cluster_id,
                    'size': len(cluster_data),
                    'missing_rates': cluster_data.isnull().mean().to_dict()
                }
                cluster_stats.append(stats)

            return pd.DataFrame(cluster_stats)
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            return pd.DataFrame()