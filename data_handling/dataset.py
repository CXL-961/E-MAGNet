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



class SIPPTimeSeriesDataset(Dataset):

    def __init__(self, data, target_columns, sequence_length=6, forecast_horizon=1):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if 'MONTHCODE' not in data.columns:
            raise ValueError("Input DataFrame must contain 'MONTHCODE' column.")

        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        self.data_df = data.copy()  # Operate on a copy

        if not pd.api.types.is_numeric_dtype(self.data_df['MONTHCODE']):
            logger.warning("SIPPTimeSeriesDataset: MONTHCODE column is not numeric, attempting conversion...")
            self.data_df['MONTHCODE'] = pd.to_numeric(self.data_df['MONTHCODE'], errors='coerce').fillna(0)

        self.scalers_sklearn = {}
        self.log_transformed_cols = []

        logger.info("SIPPTimeSeriesDataset: Starting to standardize target variables...")
        for col in self.target_columns:
            if col not in self.data_df.columns:
                logger.warning(
                    f"SIPPTimeSeriesDataset: Target column '{col}' does not exist in the data, skipping processing.")
                continue

            if col == 'RMESR':
                if not pd.api.types.is_numeric_dtype(self.data_df[col]):
                    self.data_df[col] = pd.to_numeric(self.data_df[col], errors='coerce')
                # Fill NaN RMESR with 1 (assuming 1 is a valid category or an "unknown" category)
                # Will be processed to 0-6 later in __getitem__
                if self.data_df[col].isnull().any():
                    logger.warning(
                        f"SIPPTimeSeriesDataset: Target column RMESR contains NaN, will be filled with category 1.")
                    self.data_df[col] = self.data_df[col].fillna(1.0)

                self.data_df[col] = self.data_df[col].round().astype(
                    float)  # first round then convert to float, because it will be uniformly converted to float32 later
                logger.info(f"SIPPTimeSeriesDataset: Target column RMESR processing complete.")
                continue

            # --- Process regression target columns ---
            if not pd.api.types.is_numeric_dtype(self.data_df[col]):
                logger.warning(f"SIPPTimeSeriesDataset: Target column '{col}' (type {self.data_df[col].dtype}) "
                               f"is not numeric, attempting to convert to float32...")
                self.data_df[col] = pd.to_numeric(self.data_df[col], errors='coerce').astype(np.float32)

            if self.data_df[col].isnull().sum() == len(self.data_df[col]):  # If the entire column is NaN
                logger.error(
                    f"SIPPTimeSeriesDataset: Target column '{col}' is entirely NaN, cannot standardize. Will be filled with 0.")
                self.data_df[col] = 0.0

                continue
            elif self.data_df[col].isnull().any():
                median_val = np.nanmedian(
                    self.data_df[col].astype(np.float64))  # use float64 to calculate median for better precision
                logger.warning(
                    f"SIPPTimeSeriesDataset: Target column '{col}' has NaN values ({self.data_df[col].isnull().sum()}),"
                    f" will fill with median {median_val:.4f} before standardizing.")
                self.data_df[col] = self.data_df[col].fillna(median_val)

            # TPTOTINC special handling: log(1+x) + StandardScaler
            if col == 'TPTOTINC':
                logger.info(f"SIPPTimeSeriesDataset: Applying log1p transform to target column '{col}'...")
                # clip(lower=0) to avoid taking log of negative numbers.
                self.data_df[col] = np.log1p(self.data_df[col].clip(lower=0))
                self.log_transformed_cols.append(col)

            scaler = StandardScaler()
            try:
                scaled_values = scaler.fit_transform(self.data_df[[col]].astype(np.float32))
                if scaler.scale_ is not None and np.any(np.isclose(scaler.scale_, 0)):
                    logger.warning(
                        f"SIPPTimeSeriesDataset: Standard deviation of target column '{col}' is close to 0 after standardization."
                        " This may cause issues with inverse transformation later. Original values may be constant or have very little variation.")
                self.data_df[col] = scaled_values
                self.scalers_sklearn[col] = scaler
                logger.info(f"SIPPTimeSeriesDataset: Target column '{col}' standardization complete. Log transform: "
                            f"{'Yes' if col in self.log_transformed_cols else 'No'}")
            except ValueError as e_scale:
                logger.error(f"SIPPTimeSeriesDataset: Failed to standardize target column '{col}': {e_scale}."
                             " The column may still contain non-numeric or only a single value. It will be kept as is (possibly filled with median).")

        self._ensure_features_numeric()

        self.sequences = self._create_sequences()
        if not self.sequences:
            logger.warning("SIPPTimeSeriesDataset: Failed to create any sequences. The dataset will be empty.")
        else:
            logger.info(f"SIPPTimeSeriesDataset: Successfully initialized and created {len(self.sequences)} sequences.")

    def _ensure_features_numeric(self):
        logger.debug("SIPPTimeSeriesDataset: Ensuring feature columns for X are numeric...")
        potential_feature_cols = [
            c for c in self.data_df.columns
            if c not in ['SSUID', 'PNUM', 'MONTHCODE'] + self.target_columns
        ]
        changed_cols_count = 0
        for col in potential_feature_cols:
            if col in self.data_df.columns:
                is_modified = False
                if not pd.api.types.is_numeric_dtype(self.data_df[col]):
                    original_type = self.data_df[col].dtype
                    self.data_df[col] = pd.to_numeric(self.data_df[col], errors='coerce')
                    # logger.debug(f"  Feature column '{col}': converted from {original_type} to {self.data_df[col].dtype} (due to non-numeric)")
                    is_modified = True

                if self.data_df[col].isnull().any():
                    num_na = self.data_df[col].isnull().sum()
                    self.data_df[col] = self.data_df[col].fillna(0)
                    # logger.debug(f"  Feature column '{col}': filled {num_na} NaN values with 0.")
                    is_modified = True
                if is_modified:
                    changed_cols_count += 1
        if changed_cols_count > 0:
            logger.info(
                f"SIPPTimeSeriesDataset: _ensure_features_numeric modified {changed_cols_count} feature columns (type conversion/NaN filling).")

    def _convert_numeric(self, data_df_copy):
        feature_cols_to_convert = [
            col for col in data_df_copy.columns
            if col not in ['SSUID', 'PNUM', 'MONTHCODE'] + self.target_columns
        ]

        other_target_cols_to_convert = [
            col for col in self.target_columns if col != 'RMESR'
        ]

        for col in feature_cols_to_convert + other_target_cols_to_convert:
            if col in data_df_copy.columns:
                if not pd.api.types.is_numeric_dtype(data_df_copy[col]):
                    data_df_copy[col] = pd.to_numeric(data_df_copy[col], errors='coerce')

                if col in feature_cols_to_convert:  # Only fill NaNs for feature columns here
                    data_df_copy[col] = data_df_copy[col].fillna(0)
        return data_df_copy

    def _create_sequences(self):
        sequences = []
        grouped = self.data_df.groupby(['SSUID', 'PNUM'])

        self.feature_names = []
        potential_feature_cols_for_x = [
            col for col in self.data_df.columns
            if col not in ['SSUID', 'PNUM', 'MONTHCODE'] + self.target_columns
        ]
        for col in potential_feature_cols_for_x:
            if col in self.data_df.columns and pd.api.types.is_numeric_dtype(self.data_df[col]):
                self.feature_names.append(col)

        if not self.feature_names:
            logger.error(
                "SIPPTimeSeriesDataset._create_sequences: Failed to identify any numeric feature columns. Cannot create sequences.")
            return []
        for (ssuid, pnum), group in tqdm(grouped, desc="SIPPTimeSeriesDataset: Creating sequences"):
            group = group.sort_values('MONTHCODE')
            if len(group) < self.sequence_length + self.forecast_horizon:
                continue

            x_values_all_timesteps = group[self.feature_names].values.astype(np.float32)
            y_values_all_timesteps = group[self.target_columns].values.astype(
                np.float32)  # Targets already standardized/processed

            state_col_name = 'economic_state_encoded'
            state_values_all_timesteps = group[state_col_name].values if state_col_name in group.columns else np.zeros(
                len(group), dtype=int)

            for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                x_slice_np = x_values_all_timesteps[i: i + self.sequence_length]
                y_slice_np = y_values_all_timesteps[
                             i + self.sequence_length: i + self.sequence_length + self.forecast_horizon]

                # The state is taken from the corresponding first timestep of y_slice_np
                state_at_prediction_start_time_idx = i + self.sequence_length
                state_at_prediction_time = state_values_all_timesteps[
                    state_at_prediction_start_time_idx] if state_at_prediction_start_time_idx < len(
                    state_values_all_timesteps) else 0

                if np.isnan(
                        x_slice_np).any():  # Theoretically, features are already filled with 0, this is a double check
                    x_slice_np = np.nan_to_num(x_slice_np)

                sequences.append((x_slice_np, y_slice_np, int(state_at_prediction_time)))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not self.sequences:
            raise IndexError("SIPPTimeSeriesDataset is empty or no sequences were created.")
        if idx >= len(self.sequences):
            raise IndexError(f"SIPPTimeSeriesDataset index {idx} out of range for size {len(self.sequences)}.")

        x_raw_np, y_raw_np, state_raw_int = self.sequences[idx]
        x_tensor = torch.from_numpy(x_raw_np).float()
        y_processed_np = y_raw_np.copy()

        rmesr_idx_in_targets = -1
        if 'RMESR' in self.target_columns:
            try:
                rmesr_idx_in_targets = self.target_columns.index('RMESR')
            except ValueError:
                pass

        if rmesr_idx_in_targets != -1:
            rmesr_values_at_horizon = y_processed_np[:, rmesr_idx_in_targets]
            # In __init__, NaNs in RMESR were filled with 1 and ensured to be float. Clipping and conversion are done here.
            clipped_rmesr = np.clip(np.nan_to_num(rmesr_values_at_horizon, nan=1.0).round(), 1, 7)
            y_processed_np[:, rmesr_idx_in_targets] = clipped_rmesr - 1.0  # Convert to 0-6

        y_final_tensor = torch.from_numpy(y_processed_np).float()

        if self.forecast_horizon == 1:
            y_final_tensor = y_final_tensor.squeeze(0)

        state_tensor = torch.tensor(state_raw_int, dtype=torch.long)
        return x_tensor, y_final_tensor, state_tensor

    def get_feature_names(self):
        """Get the list of feature names"""
        if hasattr(self, 'feature_names') and self.feature_names:
            return self.feature_names

        logger.warning(
            "SIPPTimeSeriesDataset.feature_names was not set correctly in _create_sequences or sequences are empty. Attempting to infer from data_df.")

        if hasattr(self, 'data_df') and isinstance(self.data_df, pd.DataFrame):
            potential_feature_cols = [
                col for col in self.data_df.columns
                if col not in ['SSUID', 'PNUM', 'MONTHCODE'] + self.target_columns
            ]
            self.feature_names = [col for col in potential_feature_cols if
                                  col in self.data_df.columns and pd.api.types.is_numeric_dtype(self.data_df[col])]
            if self.feature_names:
                logger.info(f"Successfully obtained feature names via fallback logic, count: {len(self.feature_names)}")
                return self.feature_names

        logger.error("Could not determine feature names for SIPPTimeSeriesDataset.")
        return []
    def inverse_transform_targets(self, y_pred_tensor_or_np, target_name):
        if isinstance(y_pred_tensor_or_np, torch.Tensor):
            y_pred_np = y_pred_tensor_or_np.detach().cpu().numpy()  # detach() is good practice
        else:
            y_pred_np = np.asarray(y_pred_tensor_or_np)

        if y_pred_np.size == 0: return np.array([])  # Handle empty input

        # Prepare shape for scaler (usually [N, 1])
        y_pred_np_reshaped_for_scaler = y_pred_np
        if target_name != 'RMESR':  # Regression targets need to be reshaped
            if y_pred_np.ndim == 1:
                y_pred_np_reshaped_for_scaler = y_pred_np.reshape(-1, 1)
            elif y_pred_np.ndim == 0:  # scalar
                y_pred_np_reshaped_for_scaler = np.array([[y_pred_np.item()]])
            elif y_pred_np.ndim == 2 and y_pred_np.shape[1] == 1:
                pass  # Already [N,1]
            else:  # other unexpected shapes
                logger.warning(
                    f"inverse_transform_targets for '{target_name}': Input shape {y_pred_np.shape} is not as expected, attempting to reshape.")
                try:
                    y_pred_np_reshaped_for_scaler = y_pred_np.reshape(-1, 1)
                except:  # pylint: disable=bare-except
                    logger.error(
                        f"inverse_transform_targets for '{target_name}': Reshape failed, returning original values.")
                    return y_pred_np.squeeze()

        if target_name == 'RMESR':
            # Assume y_pred_np are class indices from 0-6
            return y_pred_np + 1

        if target_name in self.scalers_sklearn:
            scaler = self.scalers_sklearn[target_name]
            if scaler.scale_ is not None and np.any(np.isclose(scaler.scale_, 0)):  # Check if standard deviation is 0
                logger.warning(
                    f"inverse_transform_targets for '{target_name}': StandardScaler's standard deviation is 0."
                    f" Inverse transform will return the mean: {scaler.mean_[0]:.4f}")
                # When scale_ is 0, inverse_transform(X) = X * 0 + mean_ = mean_
                # Therefore, all inverse-transformed values will be the mean from training time
                return np.full_like(y_pred_np_reshaped_for_scaler, scaler.mean_[0]).squeeze()

            try:
                y_pred_original_scale_np = scaler.inverse_transform(y_pred_np_reshaped_for_scaler)
            except ValueError as e_inv_scale:
                logger.error(
                    f"inverse_transform_targets for '{target_name}': Sklearn scaler inverse_transform failed: {e_inv_scale}."
                    f"Input shape: {y_pred_np_reshaped_for_scaler.shape if hasattr(y_pred_np_reshaped_for_scaler, 'shape') else 'N/A'}")
                return y_pred_np.squeeze()  # Return the pre-transformation value

            if target_name in self.log_transformed_cols:
                y_pred_original_scale_np = np.expm1(y_pred_original_scale_np)

            return y_pred_original_scale_np.squeeze()

        logger.warning(
            f"inverse_transform_targets: No corresponding sklearn scaler found for target '{target_name}'. Returning original (possibly scaled) values.")
        return y_pred_np.squeeze()
