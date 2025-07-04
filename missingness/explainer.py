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



class MissingPatternExplainerV2:
    """Structural Missing Pattern Explainer (Enhanced Version)"""

    def __init__(self, gsmd):
        self.gsmd = gsmd
        self.rule_cache = {}

    def generate_dynamic_rules(self, top_k=10):
        """Generate dynamic explanation rules (with context scope)"""
        if hasattr(self, 'cached_rules'):
            return self.cached_rules

        rules = []
        analyzer = self.gsmd.context_analyzer

        for node, node_data in analyzer.node_scores.items():
            if '_missing' not in node:
                continue

            col = node.replace('_missing', '')
            best_k = node_data['best_k']
            context_nodes = node_data.get('context_nodes', [])

            # Generate context description
            context_desc = []
            for ctx_node in context_nodes:
                if '_bin' in ctx_node:
                    cond_col, bin_val = ctx_node.split('_bin')
                    bin_val = self.gsmd._safe_bin_conversion(bin_val)
                    context_desc.append(f"{cond_col}∈{bin_val}")

            # Get original missing patterns
            patterns = self.gsmd.missing_patterns.get(col, {}).get('conditions', [])

            for pattern in sorted(patterns, key=lambda x: -x['missing_rate'])[:3]:
                rule = {
                    'target': col,
                    'context': " AND ".join(context_desc),
                    'condition': f"{pattern['condition_col']}∈{pattern['bin']}",
                    'missing_rate': pattern['missing_rate'],
                    'support': pattern['support'],
                    'stability': node_data['stability'],
                    'optimal_hops': best_k
                }
                rules.append(rule)

        rules_df = pd.DataFrame(rules).sort_values(['stability', 'missing_rate'], ascending=False)
        self.cached_rules = rules_df.head(top_k)
        return self.cached_rules

    def visualize_context_heatmap(self, node, save_path=None):
        """Visualize context heatmap"""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        G = self.gsmd.missing_graph
        analyzer = self.gsmd.context_analyzer

        if node not in analyzer.node_scores:
            raise ValueError(f"Analysis results for node {node} not found")

        best_k = analyzer.node_scores[node]['best_k']
        context_nodes = analyzer.node_scores[node]['context_nodes']

        # Create subgraph
        subgraph = G.subgraph([node] + context_nodes)
        pos = nx.spring_layout(subgraph, seed=42)

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'stability_cmap', ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Node color based on stability score
        node_colors = []
        for n in subgraph.nodes():
            if n == node:
                node_colors.append('#d7191c')  # Red indicates the target node
            else:
                stability = analyzer.node_scores.get(n, {}).get('stability', 0)
                node_colors.append(cmap(min(1, stability * 2)))

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_size=800,
            node_color=node_colors,
            alpha=0.9,
            ax=ax
        )

        # Draw weighted edges
        edges = subgraph.edges(data=True)
        edge_weights = [d['weight'] * 5 for _, _, d in edges]
        edges = nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=edges,
            width=edge_weights,
            edge_color='gray',
            alpha=0.6,
            ax=ax
        )

        # Add labels
        labels = {
            n: n.replace('_missing', '').replace('_bin', '=')
            for n in subgraph.nodes()
        }
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, ax=ax)

        # Add colorbar - now using explicit ax parameter
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Context Stability Score')

        plt.title(
            f"Missing Pattern Context for {node.replace('_missing', '')}\n"
            f"Optimal Hops: {best_k}, Stability: {analyzer.node_scores[node]['stability']:.2f}",
            fontsize=12
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_all_visualizations(self, output_dir="plots/context_heatmaps"):
        """Batch save heatmaps for all missing nodes"""
        os.makedirs(output_dir, exist_ok=True)
        missing_nodes = [
            n for n in self.gsmd.context_analyzer.node_scores
            if '_missing' in n
        ]

        for node in tqdm(missing_nodes, desc="Generating context heatmaps"):
            try:
                self.visualize_context_heatmap(
                    node,
                    save_path=os.path.join(output_dir, f"{node}_context.png"))
            except Exception as e:
                logger.error(f"Failed to generate heatmap for {node}: {str(e)}")