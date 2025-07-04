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
import torch # ensure torch is imported if used for device
from missingness.discovery import GraphStructuralMissingnessDiscovery, AdaptiveContextAnalyzerV2
from missingness.explainer import MissingPatternExplainerV2


logger = logging.getLogger(__name__)

class SIPPDataPreprocessor:

    def __init__(self, data_path, sample_size=None, robust_mode=True):
        """
        Initializes the data preprocessor

        Parameters:
            data_path (str): Path to the data file
            sample_size (int, optional): Sample size
            robust_mode (bool): Whether to enable robust processing mode (default True)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.robust_mode = robust_mode  # Controls whether to use robust processing mode
        self.raw_data = None
        self.processed_data = None
        self.missing_analyzer = None
        self.gcmp = None  # GCMP analyzer instance
        self.debug_info = []  # Storage for debugging information
        self.missing_analysis_method = 'both'  # 'tree', 'stats', or 'both'

        # Feature group definitions
        self.feature_groups = {
            'income_related': ['TPTOTINC', 'WAGP', 'INTP', 'DIVDP', 'RETP'],
            'employment_related': ['RMESR', 'WKSWORK', 'HOURS', 'OCCP', 'INDP'],
            'demographics': ['AGE', 'SEX', 'RACE', 'MARST', 'EDUC'],
            'household': ['FAMTYPE', 'FAMSIZE', 'NCHILD', 'ELDCH', 'YNGCH'],
            'government_support': ['SSIP', 'SSP', 'PAP', 'TANF', 'SNAP']
        }

        # Target and ID column definitions
        self.target_columns = ['TPTOTINC', 'TFINCPOV', 'TFINCPOVT2', 'RMESR']
        self.id_columns = ['SSUID', 'PNUM', 'MONTHCODE', 'SPANEL', 'SWAVE']

    def _analyze_missing_rates(self):
        """
        Analyzes missing rates for each column (migrated from Robust version)
        Generates a missing rate report and visualizes the Top 20
        """
        if self.raw_data is None:
            self.load_data()

        missing_rates = self.raw_data.isnull().mean().sort_values(ascending=False)
        high_missing = missing_rates[missing_rates > 0.05]  # Columns with missing rate > 5%

        logger.info("\n=== Missing Rate Analysis Results ===")
        logger.info(f"Total columns: {len(self.raw_data.columns)}")
        logger.info(f"Columns with high missing rate (>5%): {len(high_missing)}")

        if len(high_missing) > 0:
            logger.info("Top 10 columns with high missing rates:")
            for col, rate in high_missing.head(10).items():
                logger.info(f"- {col}: {rate:.2%}")

        # Visualize Top 20 missing rates
        try:
            plt.figure(figsize=(12, 6))
            missing_rates[missing_rates > 0].head(20).plot(kind='bar')
            plt.title("Top 20 Missing Rate Columns")
            plt.ylabel("Missing Rate")
            plt.xticks(rotation=45)
            plt.tight_layout()

            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/missing_rates.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Missing rate analysis plot saved to plots/missing_rates.png")
        except Exception as e:
            logger.error(f"Missing rate visualization failed: {str(e)}")

    def _visualize_missing_graph(self):
        """
        Visualizes the missing pattern causal graph (migrated from Robust version)
        Requires running analyze_missing_patterns() first to generate graph data
        """
        if not hasattr(self, 'gcmp') or self.gcmp.graph.number_of_edges() == 0:
            logger.warning("Missing graph data not generated or is empty, please run analyze_missing_patterns() first")
            return

        try:
            plt.figure(figsize=(16, 10))

            # Prepare data for plotting
            graph = self.gcmp.graph
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
            edge_weights = [d['weight' ] *15 for _, _, d in graph.edges(data=True)]

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
            # Attempting simplified plot on failure
            try:
                plt.figure(figsize=(12, 8))
                nx.draw(self.gcmp.graph, with_labels=True)
                plt.savefig('plots/missing_pattern_graph_simple.png')
                plt.close()
            except:
                logger.error("Simplified plot also failed")

    def load_data(self):
        """Loads and validates data"""
        try:
            # Read data
            self.raw_data = pd.read_feather(self.data_path)

            # Sampling
            if self.sample_size is not None:
                unique_ids = self.raw_data[['SSUID', 'PNUM']].drop_duplicates()
                sampled_ids = unique_ids.sample(min(self.sample_size, len(unique_ids)), random_state=42)
                self.raw_data = pd.merge(self.raw_data, sampled_ids, on=['SSUID', 'PNUM'])

            logger.info(f"Raw data shape: {self.raw_data.shape}")
            self._convert_dtypes()
            return self.raw_data
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _convert_dtypes(self):
        """Strict type conversion"""
        # Special handling for MONTHCODE
        if 'MONTHCODE' in self.raw_data.columns:
            self.raw_data['MONTHCODE'] = DataTypeConverter.convert_monthcode(self.raw_data['MONTHCODE'])

        # Convert other numeric columns
        numeric_cols = [col for col in self.raw_data.columns
                        if col not in self.id_columns or col == 'MONTHCODE']

        for col in numeric_cols:
            self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')

        # Convert ID columns
        for col in self.id_columns:
            if col in self.raw_data.columns and col != 'MONTHCODE':
                self.raw_data[col] = self.raw_data[col].astype(str)

        logger.info("Data type conversion validation:")
        logger.info(f"MONTHCODE type: {self.raw_data['MONTHCODE'].dtype}")
        logger.info(f"MONTHCODE example values: {self.raw_data['MONTHCODE'].head().values}")

    def analyze_missing_patterns(self, method=None):
        """Analyzes missing patterns"""
        if self.raw_data is None:
            self.load_data()

        if method is not None:
            self.missing_analysis_method = method

        self.missing_analyzer = EnhancedMissingPatternAnalyzer(self.raw_data)
        patterns = self.missing_analyzer.find_conditional_missing_patterns(
            method=self.missing_analysis_method
        )

        logger.info("Discovered conditional missing patterns (top 10):")
        # Record patterns found by decision tree
        for pattern in patterns['tree_patterns'][:10]:
            logger.info(
                f"[Decision Tree] {pattern['missing_column']} is missing when "
                f"{pattern['condition_column']}{pattern['direction']}{pattern['threshold']:.2f}"
            )

        # Record patterns found by statistical binning
        for target_col, related in patterns['stat_patterns'].items():
            for cond_col in related.keys():
                logger.info(f"[Statistical Binning] {target_col} missing is related to {cond_col} value range")

        return patterns

    def handle_missing_values(self):
        """Modified missing value handling method, only adds missingness flags"""
        if self.raw_data is None:
            self.load_data()

        if self.missing_analyzer is None:
            self.analyze_missing_patterns()

        # Only add missing flag features, do not impute actual values
        cols_to_process = [col for col in self.raw_data.columns
                           if col not in self.id_columns + self.target_columns
                           and pd.api.types.is_numeric_dtype(self.raw_data[col])]

        # Add missing flags
        for col in cols_to_process:
            self.raw_data[f'{col}_missing'] = self.raw_data[col].isnull().astype(np.float32)

        logger.info("Missing flags added (preserving original NaN values for model handling)")
        return self.raw_data

    def detect_economic_states(self, data, window_size=6):
        """Final version of economic state detection"""
        if 'TPTOTINC' not in data.columns:
            raise ValueError("TPTOTINC column does not exist")

        data = data.copy()
        data['economic_state'] = 'unknown'
        data['economic_state_encoded'] = 0

        # Ensure key columns are numeric
        data['TPTOTINC'] = pd.to_numeric(data['TPTOTINC'], errors='coerce')

        grouped = data.groupby(['SSUID', 'PNUM'])

        for (ssuid, pnum), group in grouped:
            group = group.sort_values('MONTHCODE')
            income = group['TPTOTINC'].values

            # Skip invalid data
            if len(income) < window_size or np.isnan(income).any():
                continue

            # Convert to float type array
            income = income.astype(float)
            changes = []
            volatilities = []

            for i in range(len(income) - window_size + 1):
                window = income[i: i +window_size]

                # Calculate rate of change and volatility (with safe division)
                with np.errstate(divide='ignore', invalid='ignore'):
                    change = (window[-1] - window[0]) / (abs(window[0]) + 1e-6)
                    vol = np.std(window) / (np.mean(window) + 1e-6)

                if np.isfinite(change) and np.isfinite(vol):
                    changes.append(change)
                    volatilities.append(vol)

            # Skip individuals with no valid calculations
            if not changes:
                continue

            # Convert to numpy array for vectorized calculation
            changes = np.array(changes)
            median_change = np.nanmedian(changes)

            # Safely calculate MAD
            try:
                abs_deviations = np.abs(changes - median_change)
                mad_change = 1.4826 * np.nanmedian(abs_deviations)
            except Exception as e:
                logger.warning(f"Individual ({ssuid},{pnum}) MAD calculation failed: {str(e)}")
                continue

            # Classify economic states
            states = []
            for change, vol in zip(changes, volatilities):
                if change > (median_change + 2 * mad_change):
                    states.append('growth')
                elif change < (median_change - 2 * mad_change):
                    states.append('recession')
                elif vol > 0.1:
                    states.append('volatile')
                else:
                    states.append('stable')

            # Align with original data length
            states = ['unknown'] * (window_size - 1) + states

            # Update data (ensure length matches)
            if len(states) == len(group):
                data.loc[group.index, 'economic_state'] = states
                state_mapping = {'unknown' :0, 'stable' :1, 'growth' :2, 'recession' :3, 'volatile' :4}
                data.loc[group.index, 'economic_state_encoded'] = (
                    data.loc[group.index, 'economic_state'].map(state_mapping).fillna(0).astype(int))

        return data

    def _validate_numeric_columns(self, data):
        """Validates and ensures key columns are numeric

        Parameters:
            data: DataFrame to be processed

        Returns:
            The processed DataFrame
        """
        # List of key numeric columns
        numeric_cols = [
                           'TPTOTINC', 'MONTHCODE', 'WAGP', 'INTP', 'DIVDP', 'RETP',
                           'RMESR', 'WKSWORK', 'HOURS', 'AGE'
                       ] + self.target_columns

        for col in numeric_cols:
            if col in data.columns:
                try:
                    # Attempt to convert to numeric type
                    original_type = str(data[col].dtype)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    new_type = str(data[col].dtype)

                    # Log the conversion status
                    null_count = data[col].isnull().sum()
                    if null_count > 0:
                        logger.warning(f"Column {col} type converted from {original_type} to {new_type}, "
                                       f"found {null_count} invalid values")
                    else:
                        logger.debug(f"Column {col} type converted from {original_type} to {new_type}")

                except Exception as e:
                    logger.error(f"Column {col} conversion failed: {str(e)}")
                    # If conversion fails, try filling with 0
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    logger.warning(f"Column {col} invalid values have been filled with 0")

        return data
    def _safe_economic_calculation(self, income_series, window_size=6):
        """Economic indicator calculation (returns changes and volatilities arrays)"""
        income = income_series.astype(float).values
        changes = []
        volatilities = []

        for i in range(len(income) - window_size + 1):
            window = income[i: i +window_size]

            # Skip windows containing NaN
            if np.isnan(window).any():
                continue

            # Calculate using safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                change = (window[-1] - window[0]) / (abs(window[0]) + 1e-6)
                vol = np.std(window) / (np.mean(window) + 1e-6)

            if np.isfinite(change) and np.isfinite(vol):
                changes.append(change)
                volatilities.append(vol)

        return np.array(changes), np.array(volatilities)

    def feature_engineering(self, data):
        """Feature engineering method

        Parameters:
            data: DataFrame to be processed

        Returns:
            The processed DataFrame
        """
        data = data.copy()

        # 1. Time feature processing
        if 'MONTHCODE' in data.columns:
            try:
                data['MONTHCODE'] = pd.to_numeric(data['MONTHCODE'])
                data['year'] = (data['MONTHCODE'] // 100).astype(int)
                data['month'] = (data['MONTHCODE'] % 100).astype(int)
                data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
                data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            except Exception as e:
                logger.error(f"Processing MONTHCODE failed: {str(e)}")

        # 2. Lag features
        lag_features = ['TPTOTINC', 'RMESR']
        lag_periods = [1, 3, 6]

        for feature in lag_features:
            if feature in data.columns:
                grouped = data.groupby(['SSUID', 'PNUM'])[feature]
                for lag in lag_periods:
                    new_col = f'{feature}_lag_{lag}'
                    data[new_col] = grouped.shift(lag)
                    logger.debug(f"Created lag feature: {new_col}")

        # 3. Rate of change features
        for feature in lag_features:
            for lag in [1, 3]:
                lag_col = f'{feature}_lag_{lag}'
                if lag_col in data.columns:
                    current = data[feature]
                    lagged = data[lag_col]
                    # Handle with safe division
                    with np.errstate(divide='ignore', invalid='ignore'):
                        change = (current - lagged) / (lagged.replace(0, np.nan))
                    data[f'{feature}_change_{lag}'] = change.fillna(0)

        logger.info(f"Feature engineering complete, added {len(data.columns) - len(lag_features)} features")
        return data

    def prepare_model_data(self):
        try:
            # 1. Load data
            if self.raw_data is None:
                self.load_data()

            logger.info("\n=== Starting Data Preprocessing ===")
            logger.info(f"Initial data shape: {self.raw_data.shape}")

            # 2. Validate numeric columns
            logger.info("Validating numeric columns...")
            self.raw_data = self._validate_numeric_columns(self.raw_data)

            # 3. Initialize GSMD
            logger.info("Initializing GSMD analyzer...")
            self.gsmd = GraphStructuralMissingnessDiscovery(
                data=self.raw_data,
                min_mi=0.01,
                max_bins=5,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

            # 4. Build missingness graph
            logger.info("Building missing pattern graph...")
            missing_graph = self.gsmd.build_missing_graph()
            logger.info(f"Missing graph construction complete: {missing_graph.number_of_nodes()} nodes, "
                        f"{missing_graph.number_of_edges()} edges")

            # 5. Generate GAT embeddings
            logger.info("Generating GAT multi-hop embeddings...")
            missing_embeddings = self.gsmd.generate_missing_embeddings(embedding_size=32)

            # 6. Create enhanced features
            logger.info("Creating enhanced features...")
            enhanced_data = self.gsmd.create_missing_features(self.raw_data)

            # 7. Economic state detection
            logger.info("Detecting economic states...")
            enhanced_data = self.detect_economic_states(enhanced_data)

            # 8. Feature engineering
            logger.info("Executing feature engineering...")
            enhanced_data = self.feature_engineering(enhanced_data)  # Only pass data parameter

            # 9. Final cleanup
            logger.info("Executing final cleanup...")
            final_data = self._final_cleanup(enhanced_data)

            logger.info("\n=== Data Preprocessing Complete ===")
            logger.info(f"Final data shape: {final_data.shape}")
            logger.info(f"Number of new features: {len(final_data.columns) - len(self.raw_data.columns)}")

            return final_data

        except Exception as e:
            logger.critical(f"Data preprocessing failed: {str(e)}")
            # Attempting to save intermediate results for debugging
            if hasattr(self, 'enhanced_data'):
                self.enhanced_data.to_csv('debug_failed_preprocessing.csv')
            raise


    def _final_cleanup(self, data):
        """Data cleanup"""
        # Remove temporary columns
        cols_to_drop = [col for col in data.columns
                        if col.startswith('temp_')]
        data = data.drop(columns=cols_to_drop, errors='ignore')

        # Ensure target columns have no missing values
        initial_rows = len(data)
        data = data.dropna(subset=self.target_columns)
        if len(data) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(data)} rows (due to missing target variables)")

        # Reset index
        return data.reset_index(drop=True)

    def debug_columns(self, data, n_samples=5):
        """Debug display of column types and sample values"""
        logger.debug("\n=== Data Column Debugging Information ===")
        for col in data.columns:
            logger.debug(f"{col}: {data[col].dtype} | Example: {data[col].head(n_samples).values}")



class DataTypeConverter:
    """Dedicated class for data type conversion"""
    @staticmethod
    def convert_monthcode(series):
        """Specialized handling for MONTHCODE conversion"""
        try:
            # First, try direct conversion
            converted = pd.to_numeric(series, errors='raise')
            logger.info("MONTHCODE direct conversion successful")
            return converted
        except Exception as e:
            logger.warning(f"Direct conversion of MONTHCODE failed: {str(e)}, attempting conversion after cleaning")
            # Clean non-numeric characters
            cleaned = series.str.replace(r'[^\d]', '', regex=True)
            cleaned = cleaned.replace('', '0')  # Replace empty strings with 0
            return pd.to_numeric(cleaned, errors='coerce')