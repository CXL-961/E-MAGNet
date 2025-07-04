import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import logging
from tqdm import tqdm

from models.missing_gat import GATMissingEmbedder

logger = logging.getLogger(__name__)

class GraphStructuralMissingnessDiscovery:
    """Graph-based Structural Missingness Discovery (GSMD) System"""

    def __init__(self, data, min_mi=0.01, max_bins=5, device=None):
        """
        Parameters:
            data: Input data (DataFrame)
            min_mi: Mutual information threshold
            max_bins: Maximum number of bins for continuous variables
            device: Computation device (cpu/cuda)
        """
        self.data = data.copy()
        self.min_mi = min_mi
        self.max_bins = max_bins
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.missing_graph = nx.DiGraph()
        self.missing_patterns = {}
        self.feature_bins = {}
        self.missing_embeddings = None
        self.gat_embedder = None
        self.missing_graph_pyg = None
        self.context_analyzer = AdaptiveContextAnalyzerV2(
            self.missing_graph,
            alpha=1.0,  # Adjustable
            beta=0.3,  # Adjustable
            max_hops=3
        )

        logger.info(f"GSMD initialized, will use device: {self.device}")

    def _discretize_continuous(self, col_data):
        """Robust binning for continuous variables"""
        try:
            if col_data.nunique() < 2:
                return pd.Series([0 ] *len(col_data))

            # Use equal-frequency binning to handle duplicate values
            binned = pd.qcut(
                col_data.rank(method='first'),
                q=self.max_bins,
                duplicates='drop',
                labels=False
            )
            return binned
        except Exception as e:
            logger.warning(f"Binning failed ({col_data.name}): {str(e)}, using original values")
            return col_data

    def _compute_mutual_info(self, x, y):
        """Compute mutual information (with robust handling)"""
        try:
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if valid_mask.sum() < 10:
                return 0

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            # Adaptive binning
            x_bins = min(self.max_bins, len(np.unique(x_valid)))
            y_bins = 2  # Missing indicator is binary

            if len(np.unique(x_valid)) > 10:
                x_valid = pd.cut(x_valid, bins=x_bins, labels=False)

            return mutual_info_score(x_valid, y_valid)
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {str(e)}")
            return 0

    def _visualize_missing_graph(self):
        """
        Visualize the missing pattern causal graph
        """
        if not hasattr(self, 'gcmp') or self.gcmp.graph.number_of_edges() == 0:
            logger.warning("Missing graph data not generated or is empty, please run analyze_missing_patterns() first")
            return

        try:
            plt.figure(figsize=(16, 10))

            # Prepare plotting data
            graph = self.gcmp.graph
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
            edge_weights = [d['weight'] * 15 for _, _, d in graph.edges(data=True)]

            # Draw nodes and edges
            nx.draw_networkx_nodes(
                graph, pos,
                node_size=800,
                node_color='lightblue',
                alpha=0.9
            )

            edges = nx.draw_networkx_edges(
                graph, pos,
                width=edge_weights,
                edge_color=edge_weights,
                edge_cmap=plt.cm.Blues,
                arrows=True,
                arrowstyle='->',
                arrowsize=15
            )

            # Add labels
            nx.draw_networkx_labels(
                graph, pos,
                font_size=10,
                font_family='sans-serif'
            )

            # Add color bar
            plt.colorbar(
                plt.cm.ScalarMappable(cmap=plt.cm.Blues),
                label="Mutual Information Strength",
                shrink=0.8
            )

            plt.title("Missing Pattern Causal Graph\n(Edge width/color represents association strength)", fontsize=14)
            plt.axis('off')

            os.makedirs("plots", exist_ok=True)
            plt.savefig(
                'plots/missing_pattern_graph.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            logger.info("Missing pattern graph saved to plots/missing_pattern_graph.png")

        except Exception as e:
            logger.error(f"Visualizing missing graph failed: {str(e)}")
            # Try a simplified plot on failure
            try:
                plt.figure(figsize=(12, 8))
                nx.draw(self.gcmp.graph, with_labels=True)
                plt.savefig('plots/missing_pattern_graph_simple.png')
                plt.close()
            except:
                logger.error("Simplified plot also failed")
    def build_missing_graph(self):
        """Build the missing pattern causal graph (fixes binned value handling)"""
        missing_cols = [col for col in self.data.columns if self.data[col].isnull().any()]
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()

        logger.info(f"Starting to build missing graph, analyzing {len(missing_cols)} missing columns...")

        for miss_col in tqdm(missing_cols, desc="Analyzing missing columns"):
            if self.data[miss_col].isnull().mean() < 0.03:  # Ignore columns with low missing rates
                continue

            y = self.data[miss_col].isnull().astype(int)
            self.missing_patterns[miss_col] = {'conditions': []}

            for cond_col in numeric_cols:
                if cond_col == miss_col:
                    continue

                x = self.data[cond_col]
                if x.nunique() < 2:  # Ignore constant columns
                    continue

                # Compute mutual information
                mi = self._compute_mutual_info(x, y)
                if mi < self.min_mi:
                    continue

                # Bin and compute conditional missing rates (modified bin label handling)
                x_binned = self._discretize_continuous(x)
                bin_labels = [str(float(b)) for b in sorted(x_binned.unique()) if not np.isnan(b)]

                for bin_label in bin_labels:
                    try:
                        bin_mask = (x_binned.astype(str) == bin_label)
                        missing_rate = y[bin_mask].mean()
                        total_in_bin = bin_mask.sum()

                        if missing_rate > 2 * y.mean() and total_in_bin > 10:
                            pattern = {
                                'condition_col': cond_col,
                                'bin': bin_label,  # Keep as string
                                'missing_rate': missing_rate,
                                'support': total_in_bin,
                                'mi_score': mi
                            }
                            self.missing_patterns[miss_col]['conditions'].append(pattern)

                            # Add to graph (using string form of bin_label)
                            node_name = f"{cond_col}_bin{bin_label}"
                            self.missing_graph.add_edge(
                                node_name,
                                f"{miss_col}_missing",
                                weight=missing_rate,
                                mi=mi,
                                support=total_in_bin
                            )
                    except Exception as e:
                        logger.warning(f"Processing bin {bin_label} failed: {str(e)}")
                        continue

        logger.info \
            (f"Missing graph construction complete, with {self.missing_graph.number_of_nodes()} nodes and {self.missing_graph.number_of_edges()} edges")
        self._convert_to_pyg_graph()  # Immediately convert to PyG format


        self.context_analyzer = AdaptiveContextAnalyzerV2(self.missing_graph, self.min_mi)
        # Analyze optimal hops immediately after construction
        self.optimal_hops = self.context_analyzer.find_optimal_hops()
        logger.info(f"Optimal hops analysis complete, example result: {dict(list(self.optimal_hops.items())[:3])}")
        return self.missing_graph

    def _convert_to_pyg_graph(self):
        """Convert NetworkX graph to PyG format (fixes bin value conversion issue)"""
        # Node mapping
        node_mapping = {node: i for i, node in enumerate(self.missing_graph.nodes())}

        # Edge index
        edge_index = []
        edge_attr = []
        for u, v, data in self.missing_graph.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_attr.append([data['weight']])

        # Node features (initialized as node's missing rate)
        x = torch.zeros(len(node_mapping), 1)
        for node, idx in node_mapping.items():
            if node.endswith('_missing'):
                col = node.replace('_missing', '')
                x[idx] = self.data[col].isnull().mean()
            else:
                # Handle context nodes
                if '_bin' in node:
                    try:
                        col, bin_part = node.split('_bin')

                        bin_val = float(bin_part)
                        if bin_val.is_integer():
                            bin_val = int(bin_val)
                        x[idx] = (self.data[col] == bin_val).mean()
                    except Exception as e:
                        logger.warning(f"Processing node {node} failed: {str(e)}, using default value")
                        x[idx] = 0.5  # Default value

        self.missing_graph_pyg = Data(
            x=x,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr)
        )
        logger.info(f"PyG graph conversion complete, number of nodes: {len(node_mapping)}")

    def generate_missing_embeddings(self, embedding_size=16):
        """Corrected embedding generation, ensures dimension matching"""
        if not hasattr(self, 'missing_graph_pyg'):
            self._convert_to_pyg_graph()

        pyg_data = self.missing_graph_pyg.to(self.device)

        # Dynamically determine number of layers (ensuring at least 1)
        num_layers = min(3, max(1, int(np.mean(
            [v['best_k'] for v in self.optimal_hops.values()]
        ))) if hasattr(self, 'optimal_hops') else 2)

        # Initialize GAT model
        self.gat_embedder = GATMissingEmbedder(
            embedding_size=embedding_size,
            num_heads=4,
            num_layers=num_layers
        ).to(self.device)

        # Training configuration (modified reconstructor structure)
        optimizer = torch.optim.Adam(self.gat_embedder.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        node_rates = pyg_data.x

        # Training loop (modified to auto-adapt dimensions)
        self.gat_embedder.train()
        for epoch in range(100):
            optimizer.zero_grad()
        embeddings = self.gat_embedder(pyg_data.x, pyg_data.edge_index)

        # Dynamically build reconstructor
        in_features = embeddings.size(-1)
        reconstructor = nn.Sequential(
            nn.Linear(in_features, max(8, in_features // 2)),
            nn.ReLU(),
            nn.Linear(max(8, in_features // 2), 1)
        ).to(self.device)

        reconstructed = reconstructor(embeddings)
        loss = criterion(reconstructed, node_rates)

        if epoch > 50 and '_missing' in str(self.missing_graph.nodes()):
            miss_nodes = [i for i, node in enumerate(self.missing_graph.nodes())
                          if '_missing' in node]
        loss += 0.3 * F.mse_loss(
            embeddings[miss_nodes].mean(1, keepdim=True),
            node_rates[miss_nodes]
        )

        loss.backward()
        optimizer.step()

        # Generate final embeddings
        self.gat_embedder.eval()
        with torch.no_grad():
            node_embeddings = self.gat_embedder(pyg_data.x, pyg_data.edge_index)

        # Extract missing column embeddings
        self.missing_embeddings = {
            node.replace('_missing', ''): node_embeddings[i].cpu().numpy()
            for i, node in enumerate(self.missing_graph.nodes())
            if '_missing' in node
        }

        logger.info(f"Missing embedding generation complete, dimension: {node_embeddings.shape}")
        return self.missing_embeddings

    def create_missing_features(self, data):
        """Optimized missing feature creation (resolves fragmentation and type conversion issues)

        Parameters:
            data: Original DataFrame

        Returns:
            DataFrame: Enhanced data (without fragmentation)
        """
        if not hasattr(self, 'missing_embeddings'):
            self.generate_missing_embeddings()

        # Create a copy of the data (resolves fragmentation warning)
        data = data.copy()
        missing_cols = list(self.missing_embeddings.keys())

        # 1. Pre-allocate all new features (avoids multiple inserts)
        new_features = {}

        # Basic missingness indicators
        for col in missing_cols:
            new_features[f'missing_{col}'] = data[col].isnull().astype(np.float32)

        # 2. Batch process GAT embedding features
        emb_features = {}
        ctx_features = {}

        for col, embedding in self.missing_embeddings.items():
            # Original embedding features
            for i in range(min(16, len(embedding))):  # Ensure not to exceed embedding length
                emb_features[f'missing_emb_{col}_{i}'] = (
                        new_features[f'missing_{col}'] * embedding[i]
                )

            # Multi-hop context feature groups (safely handles any embedding length)
            ctx_groups = []
            for i in range(16, len(embedding), 4):
                group_end = min( i +4, len(embedding))
                ctx_groups.append((
                    f'missing_ctx_{col}_{( i -16 )/ 4}',
                    new_features[f'missing_{col}'] * np.mean(embedding[i:group_end])
                ))

            for name, val in ctx_groups:
                ctx_features[name] = val

        # 3. Safely handle conditional interaction features
        cond_features = {}
        for col in missing_cols:
            # Get related context nodes (with error handling)
            related_nodes = []
            try:
                related_nodes = [
                    (src, data['weight'])
                    for src, _, data in self.missing_graph.in_edges(f"{col}_missing", data=True)
                ]
                related_nodes.sort(key=lambda x: -x[1])
            except Exception as e:
                logger.warning(f"Failed to get related nodes for {col}: {str(e)}")
                continue

            # Add top 3 interaction features (safely handles binned values)
            for i, (node, _) in enumerate(related_nodes[:3]):
                if '_bin' in node:
                    try:
                        cond_col, bin_part = node.split('_bin')
                        # Safely convert bin value (handles '1.0', etc.)
                        bin_val = float(bin_part)
                        if bin_val.is_integer():
                            bin_val = int(bin_val)

                        cond_feat = np.isclose(data[cond_col], bin_val).astype(np.float32)
                        cond_features[f'missing_cond_{col}_{i}'] = (
                                cond_feat * new_features[f'missing_{col}']
                        )
                    except Exception as e:
                        logger.warning(f"Failed to process conditional feature {node}: {str(e)}")

        # 4. Global missingness statistics
        total_missing = pd.DataFrame(new_features).sum(axis=1)
        global_features = {
            'global_missing_rate': total_missing / len(missing_cols),
            'global_missing_count': total_missing
        }

        # 5. Use pd.concat to merge all features at once (resolves fragmentation issue)
        all_new_features = {
            **new_features,
            **emb_features,
            **ctx_features,
            **cond_features,
            **global_features
        }

        enhanced_data = pd.concat([
            data,
            pd.DataFrame(all_new_features)
        ], axis=1)

        logger.info(f"Feature enhancement complete, added {len(all_new_features)} missingness-related features")
        return enhanced_data
    def _safe_bin_conversion(self, bin_str):
        """Safe bin value conversion (handles formats like '1.0')"""
        try:
            val = float(bin_str)
            return int(val) if val.is_integer() else val
        except ValueError:
            logger.warning(f"Could not convert bin value: {bin_str}, using original string")
            return bin_str

    def _get_related_nodes(self, missing_col):
        """Safely get related context nodes"""
        try:
            return sorted(
                [(src, data['weight'])
                 for src, _, data in self.missing_graph.in_edges(f"{missing_col}_missing", data=True)],
                key=lambda x: -x[1]
            )
        except Exception as e:
            logger.warning(f"Failed to get related nodes for {missing_col}: {str(e)}")
        return []



class AdaptiveContextAnalyzerV2:
    """Adaptive Multi-Hop Context Selector based on Information Flow Dynamics"""

    def __init__(self, graph, alpha=1.0, beta=0.5, max_hops=3):
        self.graph = graph
        self.alpha = alpha  # Information flow weight coefficient
        self.beta = beta  # Entropy regularization coefficient
        self.max_hops = max_hops
        self.node_scores = {}
        self.path_cache = {}

    def _enumerate_paths(self, node, max_depth):
        """Enumerate all paths of length <= max_depth using BFS"""
        if node in self.path_cache:
            return self.path_cache[node]

        paths = []
        queue = [(node, [node], 0)]

        while queue:
            current, path, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for pred in self.graph.predecessors(current):
                if pred not in path:  # Avoid cycles
                    new_path = path + [pred]
                    paths.append(new_path)
                    queue.append((pred, new_path, depth + 1))

        self.path_cache[node] = paths
        return paths

    def _compute_path_stability(self, path):
        """Compute the information flow strength and entropy of a single path"""
        if len(path) < 2:
            return 0, 0

        weight_product = 1.0
        mi_list = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(u, v, default={})
            weight = edge_data.get('weight', 1e-6)
            mi = edge_data.get('mi', 0)
            weight_product *= weight
            mi_list.append(mi)

        avg_mi = np.mean(mi_list) if mi_list else 0
        entropy = -weight_product * np.log(weight_product + 1e-10)

        return weight_product * avg_mi, entropy

    def evaluate_context_stability(self, node):
        """Evaluate the multi-hop context stability of a node"""
        scores = {}

        for k in range(1, self.max_hops + 1):
            total_info = 0
            total_entropy = 0
            paths = self._enumerate_paths(node, k)

            for path in paths:
                info_flow, entropy = self._compute_path_stability(path)
                total_info += info_flow
                total_entropy += entropy

            # Composite score = α*info_flow - β*entropy
            scores[k] = self.alpha * total_info - self.beta * total_entropy

        return scores

    def find_optimal_hops(self):
        """Find the optimal number of hops for all missing nodes"""
        results = {}
        missing_nodes = [n for n in self.graph.nodes() if '_missing' in n]

        for node in tqdm(missing_nodes, desc="Analyzing missing node contexts"):
            scores = self.evaluate_context_stability(node)
            best_k = max(scores, key=scores.get)

            results[node] = {
                'best_k': best_k,
                'scores': scores,
                'stability': scores[best_k],
                'context_nodes': self._get_context_nodes(node, best_k)
            }

        self.node_scores = results
        return results

    def _get_context_nodes(self, node, k):
        """Get context nodes within k hops"""
        context = set()
        queue = [(node, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= k:
                continue

            for pred in self.graph.predecessors(current):
                if pred not in context:
                    context.add(pred)
                    queue.append((pred, depth + 1))

        return list(context)