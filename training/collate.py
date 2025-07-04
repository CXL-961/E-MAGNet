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

_DEVICE_FOR_COLLATE = None
_GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW = {}
_GLOBAL_MISSING_EMB_TENSOR = None
_GLOBAL_MISSING_EMBED_SIZE = 0
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    A custom collate_fn to pass more complete graph node information.
    """
    global _DEVICE_FOR_COLLATE, _GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW, \
        _GLOBAL_MISSING_EMB_TENSOR, _GLOBAL_MISSING_EMBED_SIZE, _NUM_GSMD_MODELLED_NODES

    if _DEVICE_FOR_COLLATE is None:
        _DEVICE_FOR_COLLATE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning("collate_fn: _DEVICE_FOR_COLLATE was not set externally, auto-detected.")

    x_list, y_list, states_list = zip(*batch)

    x_batch_tensor = torch.stack(x_list).to(_DEVICE_FOR_COLLATE)
    y_batch_tensor = torch.stack(y_list).to(_DEVICE_FOR_COLLATE)
    states_batch_tensor = torch.stack(states_list).to(_DEVICE_FOR_COLLATE)

    batch_size = x_batch_tensor.shape[0]
    num_input_features = x_batch_tensor.shape[2]  # Number of features in the original input X

    # Prepare the output graph node embeddings and relevance mask
    # If _GLOBAL_MISSING_EMB_TENSOR is not initialized or is empty, create all zeros
    if _GLOBAL_MISSING_EMB_TENSOR is not None and _NUM_GSMD_MODELLED_NODES > 0 and _GLOBAL_MISSING_EMBED_SIZE > 0:
        # Each sample will see the embeddings of all GSMD graph nodes
        # batch_graph_node_embeddings_tensor shape: [batch_size, num_gsmd_modelled_nodes, missing_embed_size]
        batch_graph_node_embeddings_tensor = _GLOBAL_MISSING_EMB_TENSOR.unsqueeze(0).expand(batch_size, -1, -1)

        # batch_graph_nodes_relevance_mask shape: [batch_size, num_gsmd_modelled_nodes]
        # Initialize to all False, indicating no relevance by default
        batch_graph_nodes_relevance_mask = torch.zeros(
            batch_size,
            _NUM_GSMD_MODELLED_NODES,
            dtype=torch.bool,
            device=_DEVICE_FOR_COLLATE
        )

        for i_sample in range(batch_size):
            sample_x_features = x_batch_tensor[i_sample]  # shape: [sequence_length, num_input_features]
            for feature_idx_in_x in range(num_input_features):
                # Check if this feature index from the original input X corresponds to a node in the GSMD graph (i.e., it has a pre-computed missing embedding)
                if feature_idx_in_x in _GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW:
                    # If this feature actually has NaN in the current sample's sequence
                    if torch.isnan(sample_x_features[:, feature_idx_in_x]).any():
                        # Get the row index of this feature in _GLOBAL_MISSING_EMB_TENSOR (i.e., its index in the GSMD graph node list)
                        gsmd_node_idx = _GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW[feature_idx_in_x]
                        if 0 <= gsmd_node_idx < _NUM_GSMD_MODELLED_NODES:
                            # Mark this GSMD graph node as relevant for the current sample
                            batch_graph_nodes_relevance_mask[i_sample, gsmd_node_idx] = True
                        else:
                            logger.warning(f"collate_fn: GSMD node index {gsmd_node_idx} from mapping is out of range "
                                           f"for _NUM_GSMD_MODELLED_NODES ({_NUM_GSMD_MODELLED_NODES}).")
    else:
        # Fallback: If global GSMD embeddings are unavailable, create dummy zero embeddings and masks
        logger.warning(
            "collate_fn: _GLOBAL_MISSING_EMB_TENSOR or related global variables are invalid. Using zero graph node embeddings and masks.")
        # Determine a fallback dimension if _NUM_GSMD_MODELLED_NODES or _GLOBAL_MISSING_EMBED_SIZE is not set
        fallback_num_nodes = _NUM_GSMD_MODELLED_NODES if _NUM_GSMD_MODELLED_NODES > 0 else 1
        fallback_embed_size = _GLOBAL_MISSING_EMBED_SIZE if _GLOBAL_MISSING_EMBED_SIZE > 0 else \
            CONFIG.get("missing_embed_size", 32)  # Get fallback from CONFIG

        batch_graph_node_embeddings_tensor = torch.zeros(
            batch_size,
            fallback_num_nodes,
            fallback_embed_size,
            device=_DEVICE_FOR_COLLATE
        )
        batch_graph_nodes_relevance_mask = torch.zeros(
            batch_size,
            fallback_num_nodes,
            dtype=torch.bool,
            device=_DEVICE_FOR_COLLATE
        )

    # Return (x, y, state, all_gsmd_node_embeddings, relevance_mask_for_nodes)
    return (
        x_batch_tensor,
        y_batch_tensor,
        states_batch_tensor,
        batch_graph_node_embeddings_tensor,
        batch_graph_nodes_relevance_mask
    )
