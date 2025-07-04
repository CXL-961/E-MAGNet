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
from collections import Counter


class SHAPAnalyzer:
    def init(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)


self.device = device
self.explainer = None



def prepare_shap_data(self, dataset, sample_size=100):
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)))
    background_data = torch.stack([dataset[i][0] for i in sample_indices]).to(self.device)
    return background_data

def create_shap_explainer(self, background_data):
    try:
        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                preds, _, _ = self.model(x_tensor)
                return preds.detach().cpu().numpy()


        background_samples = background_data[:10].cpu().numpy()
        self.explainer = shap.KernelExplainer(
            model=model_predict,
            data=background_samples
        )
        return True
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {str(e)}")
        return False

def visualize_global_importance(self, dataset, feature_names, target_names, n_samples=50):
    """Visualize global feature importance - optimized version"""
    if self.explainer is None:
        logger.warning("SHAP explainer is not initialized, cannot perform analysis")
        return False

    try:
        sample_indices = np.random.choice(len(dataset), min(n_samples, len(dataset)))
        samples = torch.stack([dataset[i][0] for i in sample_indices]).cpu().numpy()

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(samples)

        # Create a chart for each target variable
        os.makedirs("plots/shap", exist_ok=True)
        for i, target in enumerate(target_names):
            plt.figure(figsize=(12, 8))

            # Handle multi-output case
            shap_val = shap_values[i] if isinstance(shap_values, list) and i < len(shap_values) else shap_values

            shap.summary_plot(
                shap_val,
                samples,
                feature_names=feature_names,
                show=False,
                plot_type="bar"
            )
            plt.title(f"Global Feature Importance for {target}")
            plt.tight_layout()
            plt.savefig(f"plots/shap/shap_global_{target}.png")
            plt.close()
            logger.info(f"Saved global SHAP plot for {target}")

        return True
    except Exception as e:
        logger.error(f"SHAP visualization failed: {str(e)}")
        return False

def visualize_economic_state_impact(self, dataset, n_samples=100):
    """Analyze the impact of economic state on predictions - revised version"""
    samples = []
    states = []

    for i in np.random.choice(len(dataset), min(n_samples, len(dataset))):
        x, _, state = dataset[i]
        samples.append(x)
        states.append(state)

    samples = torch.stack(samples).to(self.device)
    states = torch.tensor(states, dtype=torch.long)

    with torch.no_grad():
        preds, _, _ = self.model(samples)

    # Analyze prediction differences under different economic states
    state_labels = ['unknown', 'stable', 'growth', 'recession']
    results = {}

    for i, label in enumerate(state_labels):
        mask = (states == i)
        if mask.sum().item() > 0:
            state_preds = preds[mask].cpu().numpy()
            results[label] = {
                'mean': state_preds.mean(axis=0),
                'std': state_preds.std(axis=0)
            }

    # Visualize results
    plt.figure(figsize=(12, 6))
    for i, target in enumerate(CONFIG['target_cols']):
        means = [results[label]['mean'][i] for label in results if label in results]
        stds = [results[label]['std'][i] for label in results if label in results]
        labels = [label for label in results if label in results]

        plt.errorbar(
            range(len(means)),
            means,
            yerr=stds,
            label=target,
            capsize=5,
            marker='o'
        )

    plt.xticks(range(len(labels)), labels)
    plt.xlabel('Economic State')
    plt.ylabel('Predicted Value')
    plt.title('Prediction by Economic State')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/economic_state_impact.png')
    plt.close()

    return results