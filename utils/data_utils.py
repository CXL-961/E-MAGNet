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


def check_data_quality(df):
    """Comprehensive data quality check"""
    logger.info("\n=== Data Quality Check ===")

    # Basic statistics
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Total missing values: {df.isnull().sum().sum()}")

    # Column-wise analysis
    results = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / len(df)

        results.append({
            'column': col,
            'type': col_type,
            'unique': unique_count,
            'missing': missing_count,
            'missing_rate': missing_rate
        })

    # Ensure directory exists
    os.makedirs("debug", exist_ok=True)

    # Save check results
    quality_df = pd.DataFrame(results)
    quality_df.to_csv("debug/data_quality_report.csv", index=False)

    # Print problematic columns
    problem_cols = quality_df[
        (quality_df['missing_rate'] > 0.3) |
        (quality_df['unique'] == 1)
        ]

    if not problem_cols.empty:
        logger.warning("\nFound potential problem columns:")
        for _, row in problem_cols.iterrows():
            logger.warning(
                f"{row['column']}: Type={row['type']}, "
                f"Unique_values={row['unique']}, "
                f"Missing_rate={row['missing_rate']:.2%}"
            )

    return quality_df


def balance_economic_states_oversampling(dataset, target_state_counts=None, random_state=42):
    """
    Oversamples samples of different economic states in the dataset to achieve a more balanced distribution.
    :param dataset: The original SIPPTimeSeriesDataset or its Subset.
    :param target_state_counts: dict, optional, specifies the target number of samples for each state. If None, oversamples to the size of the largest class.
    :return: A ConcatDataset containing the original data and the oversampled data.
    """
    logger.info("Starting economic state balancing oversampling...")
    states_list = []
    # Need to be able to extract the state from each element of the dataset
    for i in range(len(dataset)):
        _, _, state_tensor = dataset[i]
        states_list.append(state_tensor.item())

    if not states_list:
        logger.warning("No samples in the dataset, skipping economic state oversampling.")
        return dataset

    state_counts = Counter(states_list)
    logger.info(f"Original economic state distribution: {state_counts}")

    if target_state_counts is None:
        max_count = max(state_counts.values()) if state_counts else 0
        if max_count == 0:
            logger.warning("All economic state classes are empty, cannot determine max count, skipping oversampling.")
            return dataset
        # By default, oversample all classes to the size of the largest class
        _target_state_counts = {state: max_count for state in state_counts.keys()}
    else:
        _target_state_counts = target_state_counts

    all_indices = list(range(len(dataset)))
    oversampled_indices = []

    np.random.seed(random_state)  # for reproducibility

    for state_label, current_count in state_counts.items():
        target_count_for_state = _target_state_counts.get(state_label, current_count)
        num_to_add = target_count_for_state - current_count
        if num_to_add > 0:
            state_specific_indices = [i for i, s_val in zip(all_indices, states_list) if s_val == state_label]
            if state_specific_indices:  # Ensure there are samples of this state to choose from
                oversampled_indices.extend(
                    np.random.choice(state_specific_indices, size=num_to_add, replace=True).tolist())
            else:
                logger.warning(f"Economic state {state_label} has no samples to oversample from.")

    if not oversampled_indices:
        logger.info("No samples to oversample, returning the original dataset.")
        return dataset

    logger.info(f"Will add {len(oversampled_indices)} oversampled samples.")


    final_indices = all_indices + oversampled_indices  # original + oversampled

    # Tally the new distribution
    final_states_list = [states_list[i % len(states_list)] for i in
                         final_indices]  # A bit hacky if indices can go out of bound for original states_list

    temp_final_states = []
    original_dataset_len = len(dataset)
    for final_idx in final_indices:

        if final_idx < original_dataset_len:  # Assuming all_indices are 0 to len(dataset)-1
            temp_final_states.append(states_list[final_idx])

        else:
            pass

    oversampled_subset = Subset(dataset, oversampled_indices)
    final_dataset = ConcatDataset([dataset, oversampled_subset])


    final_states_for_log = []
    for i in range(len(final_dataset)):
        _, _, state_tensor = final_dataset[i]
        final_states_for_log.append(state_tensor.item())
    logger.info(f"Economic state distribution after balanced sampling: {Counter(final_states_for_log)}")

    return final_dataset