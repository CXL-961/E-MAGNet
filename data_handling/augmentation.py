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

class AugmentedGrowthDataset(Dataset):

    def __init__(self, base_dataset, growth_indices, factor=3, noise_std=0.01):
        self.base_dataset = base_dataset
        self.samples = []

        for _ in range(factor):
            for idx in growth_indices:
                x, y, s = base_dataset[idx]
                x_aug = x + torch.randn_like(x) * noise_std
                self.samples.append((x_aug, y, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

