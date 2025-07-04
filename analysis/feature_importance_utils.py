from sipp_model import *
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
from collections import Counter #

def analyze_feature_importance(model, dataset, n_samples=100):
    model.eval()
    sample_indices = np.random.choice(len(dataset), min(n_samples, len(dataset)))
    samples = torch.stack([dataset[i][0] for i in sample_indices])


    with torch.no_grad():
        baseline_preds, _, _ = model(samples.to(device))

    importance = []
    for i in range(samples.shape[2]):
        perturbed = samples.clone()
        perturbed[:, :, i] = perturbed[:, :, i].mean()  # 用均值扰动

        with torch.no_grad():
            perturbed_preds, _, _ = model(perturbed.to(device))
            delta = F.mse_loss(baseline_preds, perturbed_preds).item()
            importance.append(delta)


    feature_names = dataset.get_feature_names() if hasattr(dataset, 'get_feature_names') \
        else [f"feat_{i}" for i in range(samples.shape[2])]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 可视化
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature',
                data=importance_df.head(20),
                palette='viridis')
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    return importance_df
