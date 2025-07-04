from torch.utils.data import DataLoader, random_split
import torch
import logging
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import json
from torch.utils.data import ConcatDataset, Subset, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import logging
from sklearn.utils.class_weight import compute_class_weight
import sys
from collections import Counter
logger = logging.getLogger(__name__)


def plot_cluster_analysis(cluster_stats, save_path="plots/missing_clusters.png"):
    """Visualize missing cluster results"""
    try:
        plt.figure(figsize=(12, 8))

        # Extract top 20 important features
        all_rates = []
        for stats in cluster_stats.to_dict('records'):
            rates = stats['missing_rates']
            all_rates.append(rates)

        rates_df = pd.DataFrame(all_rates).T
        top_features = rates_df.mean(axis=1).sort_values(ascending=False).head(20).index

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(rates_df.loc[top_features], cmap="YlGnBu", annot=True, fmt=".1f")
        plt.title("Top 20 Features Missing Rates by Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Features")
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Missing cluster analysis plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Cluster visualization failed: {str(e)}")

