import logging
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import training.collate as collate_module_fi

logger = logging.getLogger(__name__)


def _calculate_nmse(y_true, y_pred):
    """Calculates Normalized Mean Squared Error."""
    if len(y_true) < 2: return float('nan')
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    mse = np.mean((y_true_arr - y_pred_arr) ** 2)
    var_true = np.var(y_true_arr)
    if var_true < 1e-9:
        return float('inf') if mse > 1e-9 else 0.0
    return mse / var_true


def _calculate_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error (%)."""
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0

    smape_terms = np.zeros_like(denominator, dtype=float)
    mask = denominator != 0
    smape_terms[mask] = numerator[mask] / denominator[mask]

    return np.mean(smape_terms) * 100


def _calculate_ece(y_true_labels, y_pred_probs, n_bins=10):
    """Calculates Expected Calibration Error."""
    y_true_labels_arr, y_pred_probs_arr = np.asarray(y_true_labels), np.asarray(y_pred_probs)

    if len(y_true_labels_arr) == 0 or len(y_pred_probs_arr) == 0 or y_true_labels_arr.shape[0] != \
            y_pred_probs_arr.shape[0]:
        return float('nan')

    confidences = np.max(y_pred_probs_arr, axis=1)
    predictions = np.argmax(y_pred_probs_arr, axis=1)

    accuracies = (predictions == y_true_labels_arr.astype(int)).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower_bound, upper_bound = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin_cond = (confidences >= lower_bound)
        if i < n_bins - 1:
            in_bin_cond &= (confidences < upper_bound)
        else:
            in_bin_cond &= (confidences <= upper_bound)

        num_in_bin = np.sum(in_bin_cond)
        if num_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin_cond])
            confidence_in_bin = np.mean(confidences[in_bin_cond])
            ece += (num_in_bin / len(y_true_labels_arr)) * np.abs(accuracy_in_bin - confidence_in_bin)
    return ece


class ResultAnalyzer:
    """Complete result analyzer, containing all necessary attributes and methods"""

    def __init__(self, model, train_loader, val_loader, target_names, test_loader=None, device=None):
        self.model = model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_names = target_names
        self.num_rmesr_categories = model.num_rmesr_categories if hasattr(model, 'num_rmesr_categories') else 0
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None
        self.metrics = {}
        self.feature_importance = None
        self.state_labels = ['unknown', 'stable', 'growth', 'recession', 'volatile']
        self.model_trainer_instance = None

        self.dataset_instance_for_scaling = None
        temp_loader_for_ds_instance = val_loader if val_loader and hasattr(val_loader, 'dataset') and len(
            val_loader.dataset) > 0 else (test_loader if test_loader and hasattr(test_loader, 'dataset') and len(
            test_loader.dataset) > 0 else train_loader)

        if temp_loader_for_ds_instance and hasattr(temp_loader_for_ds_instance, 'dataset') and len(
                temp_loader_for_ds_instance.dataset) > 0:
            current_dataset_obj = temp_loader_for_ds_instance.dataset
            if isinstance(current_dataset_obj, torch.utils.data.Subset):
                self.dataset_instance_for_scaling = current_dataset_obj.dataset
            elif isinstance(current_dataset_obj, torch.utils.data.ConcatDataset):
                if current_dataset_obj.datasets:
                    first_ds_in_concat = current_dataset_obj.datasets[0]
                    if isinstance(first_ds_in_concat, torch.utils.data.Subset):
                        self.dataset_instance_for_scaling = first_ds_in_concat.dataset
                    else:
                        self.dataset_instance_for_scaling = first_ds_in_concat
            else:
                self.dataset_instance_for_scaling = current_dataset_obj

        if not (self.dataset_instance_for_scaling and hasattr(self.dataset_instance_for_scaling,
                                                              'inverse_transform_targets')):
            logger.error("ResultAnalyzer: Unable to get SIPPTimeSeriesDataset instance or its inverse_transform_targets method."
                         "Regression metrics will be calculated on the scaled scale, their absolute values may be difficult to interpret.")
            self.dataset_instance_for_scaling = None
        else:
            logger.info("ResultAnalyzer: Successfully obtained SIPPTimeSeriesDataset instance for inverse scaling.")

    def analyze_feature_importance(self, dataset, n_samples=100):
        try:
            base_dataset = dataset
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                if not dataset.datasets:
                    logger.error("Feature importance analysis failed: ConcatDataset is empty.")
                    return None
                base_dataset = dataset.datasets[0]
                if isinstance(base_dataset, torch.utils.data.Subset) and len(dataset.datasets) > 1:
                    logger.info("Feature importance analysis uses the first subset from ConcatDataset (usually the original training set part).")
                elif len(dataset.datasets) > 1:
                    logger.info("Feature importance analysis uses the first dataset part from ConcatDataset.")

            if not hasattr(base_dataset, '__len__') or len(base_dataset) == 0:
                logger.error("Feature importance analysis failed: Base dataset is empty.")
                return None
            sample_size = min(n_samples, len(base_dataset))
            if sample_size == 0:
                logger.error("Feature importance analysis failed: Cannot select samples (sample size is 0).")
                return None
            sample_indices = np.random.choice(len(base_dataset), sample_size, replace=False)
            samples_x_features_list = []
            for i in sample_indices:
                x_sample, _, _ = base_dataset[i]
                samples_x_features_list.append(x_sample)
            if not samples_x_features_list:
                logger.error("Feature importance analysis: Failed to extract any x feature samples from the dataset.")
                return None
            samples_x_features = torch.stack(samples_x_features_list).to(self.device)
            num_features_in_x = samples_x_features.shape[2]

            graph_node_embeddings_for_fi = None
            graph_node_relevance_mask_for_fi = None
            default_econ_state_val = 0
            economic_states_for_fi = torch.full((sample_size,), default_econ_state_val,
                                                dtype=torch.long, device=self.device)

            if hasattr(self.model, 'use_graph_attention') and self.model.use_graph_attention:
                gsmd_tensor = getattr(collate_module_fi, '_GLOBAL_MISSING_EMB_TENSOR', None)
                gsmd_nodes_count = getattr(collate_module_fi, '_NUM_GSMD_MODELLED_NODES', 0)
                gsmd_embed_size = getattr(collate_module_fi, '_GLOBAL_MISSING_EMBED_SIZE', 0)
                gsmd_map = getattr(collate_module_fi, '_GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW', {})

                if gsmd_tensor is not None and gsmd_nodes_count > 0 and gsmd_embed_size > 0:
                    graph_node_embeddings_for_fi = gsmd_tensor.unsqueeze(0).expand(sample_size, -1, -1).clone().to(
                        self.device)
                    graph_node_relevance_mask_for_fi = torch.zeros(sample_size, gsmd_nodes_count, dtype=torch.bool,
                                                                   device=self.device)
                    for i_samp_fi in range(sample_size):
                        sample_x_seq = samples_x_features[i_samp_fi]
                        for feat_idx_in_x in range(num_features_in_x):
                            if feat_idx_in_x in gsmd_map:
                                if torch.isnan(sample_x_seq[:, feat_idx_in_x]).any():
                                    gsmd_node_idx_fi = gsmd_map[feat_idx_in_x]
                                    if 0 <= gsmd_node_idx_fi < gsmd_nodes_count:
                                        graph_node_relevance_mask_for_fi[i_samp_fi, gsmd_node_idx_fi] = True
                else:
                    logger.warning("FI: Global GSMD embedding information is insufficient or unavailable. Graph attention will use dummy inputs.")
                    model_gsmd_embed_size = getattr(self.model, 'gsmd_node_embed_size', 32)
                    num_dummy_nodes_fi = 1
                    graph_node_embeddings_for_fi = torch.zeros(sample_size, num_dummy_nodes_fi, model_gsmd_embed_size,
                                                               device=self.device)
                    graph_node_relevance_mask_for_fi = torch.ones(sample_size, num_dummy_nodes_fi, dtype=torch.bool,
                                                                  device=self.device)
            self.model.eval()
            with torch.no_grad():
                baseline_preds, _, _, _ = self.model(
                    samples_x_features,
                    economic_state=economic_states_for_fi,
                    batch_graph_node_embeddings=graph_node_embeddings_for_fi,
                    batch_graph_nodes_relevance_mask=graph_node_relevance_mask_for_fi,
                    prev_state_lstm=None
                )
            importance_scores = []
            for i in range(num_features_in_x):
                perturbed_x = samples_x_features.clone()
                idx_perm = torch.randperm(perturbed_x.size(0))
                perturbed_x[:, :, i] = perturbed_x[idx_perm, :, i]
                with torch.no_grad():
                    perturbed_preds, _, _, _ = self.model(
                        perturbed_x,
                        economic_state=economic_states_for_fi,
                        batch_graph_node_embeddings=graph_node_embeddings_for_fi,
                        batch_graph_nodes_relevance_mask=graph_node_relevance_mask_for_fi,
                        prev_state_lstm=None
                    )
                    delta = F.mse_loss(baseline_preds, perturbed_preds, reduction='mean').item()
                    importance_scores.append(delta)
            feature_names = self._get_feature_names(base_dataset)
            if not feature_names or len(feature_names) != num_features_in_x:
                logger.warning(
                    f"Number of feature names ({len(feature_names if feature_names else [])}) does not match or was not obtained for the actual number of features ({num_features_in_x}) "
                    f"Will use generic names."
                )
                feature_names = [f"feature_{j}" for j in range(num_features_in_x)]
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            self._plot_feature_importance()
            return self.feature_importance
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {str(e)}", exc_info=True)
            self.feature_importance = None
            return None

    def _get_feature_names(self, dataset_obj_for_names):
        current_ds = dataset_obj_for_names
        max_depth = 5
        count = 0
        while count < max_depth:
            if isinstance(current_ds, torch.utils.data.Subset):
                current_ds = current_ds.dataset
            elif isinstance(current_ds, torch.utils.data.ConcatDataset):
                if not current_ds.datasets:
                    logger.warning("_get_feature_names: ConcatDataset is empty.")
                    break
                current_ds = current_ds.datasets[0]
            else:
                break
            count += 1
        if count == max_depth: logger.warning("_get_feature_names: Reached max depth.")
        if hasattr(current_ds, 'get_feature_names') and callable(current_ds.get_feature_names):
            return current_ds.get_feature_names()
        elif hasattr(current_ds, 'feature_names') and current_ds.feature_names:
            return current_ds.feature_names
        elif hasattr(current_ds, '__len__') and len(current_ds) > 0 and hasattr(current_ds, '__getitem__'):
            try:
                sample_x_feat, _, _ = current_ds[0]
                if isinstance(sample_x_feat, torch.Tensor) and sample_x_feat.ndim == 2:
                    num_feat = sample_x_feat.shape[1]
                    logger.warning(f"_get_feature_names: Fallback to inferring from sample, num_features={num_feat}.")
                    return [f"feat_{i}" for i in range(num_feat)]
            except Exception as e_get_item:
                logger.warning(f"_get_feature_names: Error getting sample for fallback: {e_get_item}")
        logger.error("_get_feature_names: Could not determine feature names.")
        return []

    def _plot_feature_importance(self):
        if self.feature_importance is None or self.feature_importance.empty:
            logger.warning("Cannot plot feature importance: data is empty.")
            return
        try:
            os.makedirs("plots", exist_ok=True)
            plt.figure(figsize=(12, max(8, len(self.feature_importance.head(20)) * 0.5)))
            sns.barplot(x='importance', y='feature', data=self.feature_importance.head(20), palette='viridis')
            plt.title("Top 20 Feature Importances (Permutation-based)")
            plt.xlabel("Importance (Increase in MSE after Permutation)")
            plt.ylabel("Feature")
            plt.grid(axis='x', linestyle=':', alpha=0.6)
            plt.tight_layout()
            save_path = 'plots/feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance plot saved as {save_path}")
        except ImportError:
            logger.error("Seaborn or Matplotlib not installed, cannot plot feature importance.")
        except Exception as e:
            logger.error(f"Feature importance visualization failed: {e}", exc_info=True)

    def _analyze_state_impact(self, loader):
        self.model.eval()
        all_preds_raw_model_output_list, all_targets_dataset_processed_list, all_states_list = [], [], []
        if not (loader and hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__') and len(
                loader.dataset) > 0):
            logger.warning("ResultAnalyzer._analyze_state_impact: loader/dataset is empty. Returning empty dict.")
            return {}
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                try:
                    x_b, y_b_proc, s_b, batch_graph_node_embeddings_b, batch_graph_nodes_relevance_mask_b = batch_data
                    preds_raw_model, _, _, _ = self.model(
                        x_b.to(self.device),
                        economic_state=s_b.to(self.device),
                        batch_graph_node_embeddings=batch_graph_node_embeddings_b.to(self.device),
                        batch_graph_nodes_relevance_mask=batch_graph_nodes_relevance_mask_b.to(self.device),
                        prev_state_lstm=None
                    )
                    all_preds_raw_model_output_list.append(preds_raw_model.cpu())
                    all_targets_dataset_processed_list.append(y_b_proc.cpu())
                    all_states_list.append(s_b.cpu())
                except Exception as e_batch_state:
                    logger.error(f"ResultAnalyzer._analyze_state_impact: Error processing batch {batch_idx}: {e_batch_state}",
                                 exc_info=True)
                    continue
        if not all_preds_raw_model_output_list:
            logger.warning("ResultAnalyzer._analyze_state_impact: No predictions generated. Returning empty dict.")
            return {}
        preds_raw_model_np = torch.cat(all_preds_raw_model_output_list, dim=0).numpy()
        targets_dataset_proc_np = torch.cat(all_targets_dataset_processed_list, dim=0).numpy()
        if all_states_list:
            temp_states_tensor = torch.cat(all_states_list, dim=0)
            states_np = temp_states_tensor.numpy()
            if temp_states_tensor.ndim > 1: states_np = states_np.squeeze()
            if states_np.ndim == 0: states_np = np.array([states_np.item()])
        else:
            states_np = np.empty((0,))
        results_by_state = {}
        model_output_sizes = getattr(self.model, 'output_sizes', [])
        if not model_output_sizes:
            logger.error("_analyze_state_impact: self.model.output_sizes is not set or empty. Cannot proceed.")
            return {}
        for state_numeric_val, state_label_str in enumerate(self.state_labels):
            mask_this_state = (states_np == state_numeric_val) if states_np.size > 0 else np.array([], dtype=bool)
            if not mask_this_state.any():
                results_by_state[state_label_str] = {
                    tn: {'error': float('nan'), 'pred_mean': float('nan'), 'target_mean': float('nan')} for tn in
                    self.target_names
                }
                if 'RMESR' in self.target_names:
                    results_by_state[state_label_str]['RMESR']['pred_mode'] = float('nan')
                    results_by_state[state_label_str]['RMESR']['target_mean_or_mode'] = float('nan')
                continue
            state_specific_results_dict = {}
            current_pred_idx_in_raw_model_inner = 0
            for i_target_model, target_name_model in enumerate(self.target_names):
                if i_target_model >= len(model_output_sizes):
                    logger.error(
                        f"_analyze_state_impact: Target index {i_target_model} ('{target_name_model}') OOB for model_output_sizes. Skipping.")
                    continue
                num_outputs_this_head_model = model_output_sizes[i_target_model]
                preds_this_head_this_state_raw = preds_raw_model_np[mask_this_state,
                                                 current_pred_idx_in_raw_model_inner: current_pred_idx_in_raw_model_inner + num_outputs_this_head_model]
                targets_this_target_this_state_proc = targets_dataset_proc_np[mask_this_state, i_target_model]
                is_rmesr_model_head = (target_name_model == 'RMESR' and getattr(self.model, 'num_rmesr_categories',
                                                                                0) > 0 and num_outputs_this_head_model == self.model.num_rmesr_categories)
                target_specific_metrics_for_state = {}
                if is_rmesr_model_head:
                    pred_classes_0_6_this_state = np.argmax(preds_this_head_this_state_raw, axis=1)
                    true_classes_0_6_this_state = targets_this_target_this_state_proc.astype(int)
                    error_val = 1.0 - accuracy_score(true_classes_0_6_this_state, pred_classes_0_6_this_state) if len(
                        true_classes_0_6_this_state) > 0 else float('nan')
                    pred_mode_0_6 = -1.0
                    if len(pred_classes_0_6_this_state) > 0:
                        unique_cls, counts_cls = np.unique(pred_classes_0_6_this_state, return_counts=True)
                        pred_mode_0_6 = float(unique_cls[counts_cls.argmax()])
                    pred_mode_display = pred_mode_0_6 + 1.0 if pred_mode_0_6 != -1.0 else float('nan')
                    target_mode_display = float('nan')
                    if len(true_classes_0_6_this_state) > 0:
                        unique_true_cls, counts_true_cls = np.unique(true_classes_0_6_this_state, return_counts=True)
                        true_mode_0_6 = float(unique_true_cls[counts_true_cls.argmax()])
                        target_mode_display = true_mode_0_6 + 1.0
                    if self.dataset_instance_for_scaling:
                        try:
                            if pred_mode_0_6 != -1.0: pred_mode_display = self.dataset_instance_for_scaling.inverse_transform_targets(
                                np.array([pred_mode_0_6]), target_name_model).item()
                            if not np.isnan(
                                    target_mode_display): target_mode_display = self.dataset_instance_for_scaling.inverse_transform_targets(
                                np.array([true_mode_0_6]), target_name_model).item()
                        except Exception as e_inv_state_rmesr:
                            logger.warning(
                                f"State impact RMESR inv transform failed: {e_inv_state_rmesr}. Using +1 shift.")
                    target_specific_metrics_for_state = {'error': float(error_val),
                                                         'pred_mode': float(pred_mode_display),
                                                         'target_mean_or_mode': float(target_mode_display)}
                else:
                    pred_vals_scaled_this_state = preds_this_head_this_state_raw.squeeze(
                        -1) if preds_this_head_this_state_raw.ndim > 1 and preds_this_head_this_state_raw.shape[
                        -1] == 1 else preds_this_head_this_state_raw
                    true_vals_scaled_this_state = targets_this_target_this_state_proc
                    error_val_regr = mean_squared_error(true_vals_scaled_this_state,
                                                        pred_vals_scaled_this_state) if len(
                        true_vals_scaled_this_state) > 0 else float('nan')
                    pred_mean_display, pred_std_display, target_mean_display_regr = float('nan'), float('nan'), float(
                        'nan')
                    if len(pred_vals_scaled_this_state) > 0:
                        if self.dataset_instance_for_scaling:
                            try:
                                pred_vals_inv = self.dataset_instance_for_scaling.inverse_transform_targets(
                                    pred_vals_scaled_this_state, target_name_model)
                                true_vals_inv = self.dataset_instance_for_scaling.inverse_transform_targets(
                                    true_vals_scaled_this_state, target_name_model)
                                pred_mean_display, pred_std_display, target_mean_display_regr = np.mean(
                                    pred_vals_inv), np.std(pred_vals_inv), np.mean(true_vals_inv)
                            except Exception as e_inv_st_regr:
                                logger.warning(
                                    f"State impact Regr '{target_name_model}' inv transform failed: {e_inv_st_regr}. Using scaled.")
                                pred_mean_display, pred_std_display, target_mean_display_regr = np.mean(
                                    pred_vals_scaled_this_state), np.std(pred_vals_scaled_this_state), np.mean(
                                    true_vals_scaled_this_state)
                        else:
                            pred_mean_display, pred_std_display, target_mean_display_regr = np.mean(
                                pred_vals_scaled_this_state), np.std(pred_vals_scaled_this_state), np.mean(
                                true_vals_scaled_this_state)
                    target_specific_metrics_for_state = {'error': float(error_val_regr),
                                                         'pred_mean': float(pred_mean_display),
                                                         'pred_std': float(pred_std_display),
                                                         'target_mean': float(target_mean_display_regr)}
                state_specific_results_dict[target_name_model] = target_specific_metrics_for_state
                current_pred_idx_in_raw_model_inner += num_outputs_this_head_model
            results_by_state[state_label_str] = state_specific_results_dict
        return results_by_state

    def set_training_history(self, history):
        self.history = {
            'train_loss': history.get('train_loss', history.get('train', [])),
            'val_loss': history.get('val_loss', history.get('val', []))
        }
        if not self.history['train_loss']: logger.warning("Training history (train loss) is empty or in incorrect format.")
        if not self.history['val_loss']: logger.warning("Training history (validation loss) is empty or in incorrect format.")

    def _predict_loader(self, loader):
        self.model.eval()
        all_preds_raw_list, all_targets_processed_list, all_states_list = [], [], []

        model_output_sizes = getattr(self.model, 'output_sizes', [])
        default_output_dim = sum(model_output_sizes) if model_output_sizes else len(self.target_names)
        default_target_dim = len(self.target_names)

        if not (loader and hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__') and len(
                loader.dataset) > 0):
            logger.warning("ResultAnalyzer._predict_loader: loader/dataset is empty. Returning empty arrays.")
            # Return 5 items as per the new signature
            return (np.empty((0, default_output_dim)), np.empty((0, default_target_dim)),
                    np.empty((0, default_output_dim)), np.empty((0, default_target_dim)),
                    np.empty((0,)))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                try:
                    x_batch, y_batch_processed, states_batch, \
                        batch_graph_node_embeddings, batch_graph_nodes_relevance_mask = batch_data

                    preds_raw_batch, _, _, _ = self.model(
                        x_batch.to(self.device),
                        economic_state=states_batch.to(self.device),
                        batch_graph_node_embeddings=batch_graph_node_embeddings.to(self.device),
                        batch_graph_nodes_relevance_mask=batch_graph_nodes_relevance_mask.to(self.device),
                        prev_state_lstm=None
                    )
                    all_preds_raw_list.append(preds_raw_batch.cpu())
                    all_targets_processed_list.append(
                        y_batch_processed.cpu())  # These are 0-6 for RMESR, scaled for regr
                    all_states_list.append(states_batch.cpu())
                except Exception as e_batch:
                    logger.error(f"ResultAnalyzer._predict_loader: Error processing batch {batch_idx}: {e_batch}",
                                 exc_info=True)
                    continue

        if not all_preds_raw_list:
            logger.warning("ResultAnalyzer._predict_loader: all_preds_raw_list is empty after model prediction. Returning empty arrays.")
            return (np.empty((0, default_output_dim)), np.empty((0, default_target_dim)),
                    np.empty((0, default_output_dim)), np.empty((0, default_target_dim)),
                    np.empty((0,)))

        preds_combined_raw_np = torch.cat(all_preds_raw_list, dim=0).numpy()
        targets_combined_processed_np = torch.cat(all_targets_processed_list, dim=0).numpy()

        states_np = torch.cat(all_states_list, dim=0).numpy() if all_states_list else np.empty((0,))
        if states_np.ndim > 1: states_np = states_np.squeeze()
        if states_np.ndim == 0 and states_np.size == 1: states_np = np.array([states_np.item()])

        _final_preds_cols_for_metrics = []
        _final_targets_cols_for_metrics = []
        current_pred_idx_in_raw = 0

        if not model_output_sizes:
            logger.warning("_predict_loader: model.output_sizes not found. Using defaults which might be incorrect.")
            if preds_combined_raw_np.shape[1] > 0 and len(self.target_names) > 0:
                model_output_sizes = [preds_combined_raw_np.shape[1] // len(self.target_names)] * len(self.target_names)
                if sum(model_output_sizes) != preds_combined_raw_np.shape[1]:
                    logger.error("Fallback for model_output_sizes failed. Metrics will be unreliable.")
                    model_output_sizes = [1] * len(self.target_names)
            else:
                model_output_sizes = [1] * len(self.target_names)

        for i_col_dataset, target_name_loop in enumerate(self.target_names):
            if i_col_dataset >= len(model_output_sizes):
                logger.error(
                    f"_predict_loader: Target index {i_col_dataset} ('{target_name_loop}') OOB for model_output_sizes. Skipping.")
                nan_col = np.full((preds_combined_raw_np.shape[0], 1), np.nan)
                _final_preds_cols_for_metrics.append(nan_col)
                _final_targets_cols_for_metrics.append(nan_col)
                continue

            num_outputs_for_this_head = model_output_sizes[i_col_dataset]
            y_pred_raw_this_head_np = preds_combined_raw_np[:,
                                      current_pred_idx_in_raw: current_pred_idx_in_raw + num_outputs_for_this_head]
            y_true_processed_this_target_np = targets_combined_processed_np[:, i_col_dataset]

            current_final_pred_col = None
            current_final_target_col = None

            is_rmesr_current_target = (
                    target_name_loop == 'RMESR' and self.num_rmesr_categories > 0 and num_outputs_for_this_head == self.num_rmesr_categories)

            if is_rmesr_current_target:
                pred_classes_0_6 = np.argmax(y_pred_raw_this_head_np, axis=1)
                if self.dataset_instance_for_scaling:
                    try:
                        current_final_pred_col = self.dataset_instance_for_scaling.inverse_transform_targets(
                            pred_classes_0_6, target_name_loop)
                        current_final_target_col = self.dataset_instance_for_scaling.inverse_transform_targets(
                            y_true_processed_this_target_np, target_name_loop)
                    except Exception as e_inv_rmesr:
                        logger.warning(
                            f"RMESR inv_transform failed for {target_name_loop}: {e_inv_rmesr}. Using 0-6 + 1 for display.")
                        current_final_pred_col = pred_classes_0_6 + 1
                        current_final_target_col = y_true_processed_this_target_np.astype(int) + 1
                else:
                    current_final_pred_col = pred_classes_0_6 + 1
                    current_final_target_col = y_true_processed_this_target_np.astype(int) + 1
            else:
                pred_scaled_regr = y_pred_raw_this_head_np.squeeze(-1) if y_pred_raw_this_head_np.ndim > 1 and \
                                                                          y_pred_raw_this_head_np.shape[
                                                                              -1] == 1 else y_pred_raw_this_head_np
                if self.dataset_instance_for_scaling:
                    try:
                        current_final_pred_col = self.dataset_instance_for_scaling.inverse_transform_targets(
                            pred_scaled_regr, target_name_loop)
                        current_final_target_col = self.dataset_instance_for_scaling.inverse_transform_targets(
                            y_true_processed_this_target_np, target_name_loop)
                    except Exception as e_inv_regr:
                        logger.error(
                            f"Error inv-transforming regr target '{target_name_loop}': {e_inv_regr}. Using scaled.")
                        current_final_pred_col = pred_scaled_regr
                        current_final_target_col = y_true_processed_this_target_np
                else:
                    current_final_pred_col = pred_scaled_regr
                    current_final_target_col = y_true_processed_this_target_np

            _final_preds_cols_for_metrics.append(np.asarray(current_final_pred_col).reshape(-1, 1))
            _final_targets_cols_for_metrics.append(np.asarray(current_final_target_col).reshape(-1, 1))
            current_pred_idx_in_raw += num_outputs_for_this_head

        if not _final_preds_cols_for_metrics or not _final_targets_cols_for_metrics:
            logger.error("_predict_loader: Failed to populate final metric columns after loop.")
            final_preds_for_metrics_np = np.empty((preds_combined_raw_np.shape[0], 0))
            final_targets_for_metrics_np = np.empty((preds_combined_raw_np.shape[0], 0))
        else:
            final_preds_for_metrics_np = np.concatenate(_final_preds_cols_for_metrics, axis=1)
            final_targets_for_metrics_np = np.concatenate(_final_targets_cols_for_metrics, axis=1)

        return (final_preds_for_metrics_np, final_targets_for_metrics_np,
                preds_combined_raw_np, targets_combined_processed_np,
                states_np)

    def calculate_all_metrics(self):
        try:
            train_metrics = {}
            if self.train_loader and hasattr(self.train_loader, 'dataset') and len(self.train_loader.dataset) > 0:
                train_metrics = self._calculate_metrics(self.train_loader)
            else:
                logger.warning("Training loader is empty. Skipping train metrics calculation.")

            val_metrics = {}
            state_impact_analysis = {}
            if self.val_loader and hasattr(self.val_loader, 'dataset') and len(self.val_loader.dataset) > 0:
                val_metrics = self._calculate_metrics(self.val_loader)
                state_impact_analysis = self._analyze_state_impact(self.val_loader)
            else:
                logger.warning("Validation loader is empty. Skipping validation metrics and state impact analysis.")

            test_metrics = {}
            if self.test_loader and hasattr(self.test_loader, 'dataset') and len(self.test_loader.dataset) > 0:
                logger.info("Calculating metrics on the test set...")
                test_metrics = self._calculate_metrics(self.test_loader)
            else:
                logger.warning("Test loader is empty or not provided. Skipping test metrics calculation.")


            self.metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'state_impact': state_impact_analysis
            }
            return self._sanitize_metrics(self.metrics)
        except Exception as e:
            logger.error(f"calculate_all_metrics failed: {e}", exc_info=True)
            return {'train': {}, 'val': {}, 'test': {}, 'state_impact': {}}

    def _sanitize_metrics(self, metrics):
        if isinstance(metrics, dict):
            return {k: self._sanitize_metrics(v) for k, v in metrics.items()}
        elif isinstance(metrics, (list, tuple)):
            return [self._sanitize_metrics(x) for x in metrics]
        elif isinstance(metrics, np.generic):
            return metrics.item()
        elif isinstance(metrics, (int, float, str, bool)) or metrics is None:
            return metrics
        elif torch.is_tensor(metrics):
            return metrics.item() if metrics.numel() == 1 else metrics.tolist()
        else:
            logger.warning(f"Unrecognized type in _sanitize_metrics: {type(metrics)}. Converting to string.")
            return str(metrics)

    def _calculate_metrics(self, loader):
        final_preds_inv, final_targets_inv, raw_model_outputs_all, processed_targets_all, _ = self._predict_loader(
            loader)

        metrics_result = {}
        model_output_sizes = getattr(self.model, 'output_sizes', [])
        if not model_output_sizes:  # Fallback
            model_output_sizes = [1] * len(self.target_names)
            logger.warning(
                "_calculate_metrics: model.output_sizes not found. Using default [1] per target. This may be incorrect.")

        if final_preds_inv.shape[0] == 0:
            logger.warning(
                f"ResultAnalyzer._calculate_metrics: final_preds_inv array is empty for loader. Cannot compute metrics.")
            # Initialize with all potential new metrics
            for i_name, name_ in enumerate(self.target_names):
                is_rmesr_init = (name_ == 'RMESR' and self.num_rmesr_categories > 0 and
                                 i_name < len(model_output_sizes) and model_output_sizes[
                                     i_name] == self.num_rmesr_categories)
                if is_rmesr_init:
                    metrics_result[name_] = {'accuracy': float('nan'), 'f1': float('nan'), 'ece': float('nan')}
                else:
                    metrics_result[name_] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
                                             'nmse': float('nan'), 'smape': float('nan'),
                                             'accuracy_bin': float('nan'), 'f1_bin': float('nan'),
                                             'threshold': float('nan')}
            return metrics_result

        current_raw_pred_idx = 0  # To slice raw_model_outputs_all for ECE
        for i_col, target_name in enumerate(self.target_names):
            y_true_col_inv = final_targets_inv[:, i_col]
            y_pred_col_inv = final_preds_inv[:, i_col]

            valid_indices = np.isfinite(y_true_col_inv) & np.isfinite(y_pred_col_inv)
            y_true_col_inv_filt = y_true_col_inv[valid_indices]
            y_pred_col_inv_filt = y_pred_col_inv[valid_indices]

            num_outputs_for_this_head = model_output_sizes[i_col] if i_col < len(model_output_sizes) else 1

            if len(y_true_col_inv_filt) == 0:
                logger.warning(f"_calculate_metrics for '{target_name}': Array empty after removing NaNs. Skipping.")
                is_rmesr_nan_case = (target_name == 'RMESR' and self.num_rmesr_categories > 0 and
                                     i_col < len(model_output_sizes) and model_output_sizes[
                                         i_col] == self.num_rmesr_categories)
                if is_rmesr_nan_case:
                    metrics_result[target_name] = {'accuracy': float('nan'), 'f1': float('nan'), 'ece': float('nan')}
                else:
                    metrics_result[target_name] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
                                                   'nmse': float('nan'), 'smape': float('nan'),
                                                   'accuracy_bin': float('nan'), 'f1_bin': float('nan'),
                                                   'threshold': float('nan')}
                current_raw_pred_idx += num_outputs_for_this_head
                continue

            is_rmesr_target = (target_name == 'RMESR' and self.num_rmesr_categories > 0 and
                               i_col < len(model_output_sizes) and model_output_sizes[
                                   i_col] == self.num_rmesr_categories)

            if is_rmesr_target:
                y_true_rmesr_int = y_true_col_inv_filt.astype(int)
                y_pred_rmesr_int = y_pred_col_inv_filt.astype(int)
                metrics_result[target_name] = {
                    'accuracy': accuracy_score(y_true_rmesr_int, y_pred_rmesr_int),
                    'f1': f1_score(y_true_rmesr_int, y_pred_rmesr_int, average='macro', zero_division=0)
                }
                ece_val = float('nan')
                rmesr_logits_for_ece = raw_model_outputs_all[valid_indices,
                                       current_raw_pred_idx: current_raw_pred_idx + num_outputs_for_this_head]
                rmesr_true_labels_0_6_for_ece = processed_targets_all[valid_indices, i_col].astype(int)

                if rmesr_logits_for_ece.shape[0] > 0:
                    rmesr_probs_for_ece = torch.softmax(torch.from_numpy(rmesr_logits_for_ece).float(), dim=1).numpy()
                    ece_val = _calculate_ece(rmesr_true_labels_0_6_for_ece, rmesr_probs_for_ece)
                metrics_result[target_name]['ece'] = ece_val
            else:
                mse_val = mean_squared_error(y_true_col_inv_filt, y_pred_col_inv_filt)
                mae_val = mean_absolute_error(y_true_col_inv_filt, y_pred_col_inv_filt)
                r2_val = float('nan')
                if len(y_true_col_inv_filt) >= 2 and np.var(y_true_col_inv_filt) > 1e-9:
                    r2_val = r2_score(y_true_col_inv_filt, y_pred_col_inv_filt)

                nmse_val = _calculate_nmse(y_true_col_inv_filt, y_pred_col_inv_filt)
                smape_val = _calculate_smape(y_true_col_inv_filt, y_pred_col_inv_filt)

                threshold_val_reg = np.median(y_true_col_inv_filt) if len(np.unique(y_true_col_inv_filt)) > 1 else (
                    y_true_col_inv_filt[0] if len(y_true_col_inv_filt) > 0 else 0.5)
                bin_true_reg = (y_true_col_inv_filt > threshold_val_reg).astype(int)
                bin_pred_reg = (y_pred_col_inv_filt > threshold_val_reg).astype(int)
                acc_val_reg_bin = accuracy_score(bin_true_reg, bin_pred_reg) if len(bin_true_reg) > 0 else float('nan')
                f1_val_reg_bin = f1_score(bin_true_reg, bin_pred_reg, average='binary', zero_division=0) if len(
                    bin_true_reg) > 0 else float('nan')

                metrics_result[target_name] = {
                    'mse': mse_val, 'mae': mae_val, 'r2': r2_val,
                    'nmse': nmse_val, 'smape': smape_val,
                    'accuracy_bin': acc_val_reg_bin, 'f1_bin': f1_val_reg_bin,
                    'threshold': float(threshold_val_reg)
                }
            current_raw_pred_idx += num_outputs_for_this_head
        return metrics_result

    def _get_target_unit(self, target_name):
        if target_name == 'TPTOTINC':
            return "Dollars"
        elif target_name in ['TFINCPOV', 'TFINCPOVT2']:
            return "Points"
        elif target_name == 'RMESR':
            return "Level"
        return ""

    def _plot_state_impact(self, state_impact_data_to_plot):
        if not state_impact_data_to_plot:
            logger.warning("Cannot plot economic state impact: data is empty.")
            return
        os.makedirs("plots", exist_ok=True)

        # Plot 1
        plt.figure(figsize=(14, 7))
        plot_any_mean = False
        for i, target in enumerate(self.target_names):
            means, stds, plot_states_mean = [], [], []
            for state_label_plot in self.state_labels:
                state_data = state_impact_data_to_plot.get(state_label_plot, {})
                target_data = state_data.get(target, {})
                mean_key = 'pred_mode' if target == 'RMESR' and 'pred_mode' in target_data else 'pred_mean'
                std_key = 'pred_std'
                current_mean = target_data.get(mean_key)
                if current_mean is not None and not np.isnan(current_mean):
                    means.append(current_mean)
                    current_std = target_data.get(std_key, 0) if target != 'RMESR' else 0
                    stds.append(current_std if not np.isnan(current_std) else 0)
                    plot_states_mean.append(state_label_plot)
                    plot_any_mean = True
            if means:
                plt.errorbar(plot_states_mean, means, yerr=stds if target != 'RMESR' else None,
                             label=f"{target} ({self._get_target_unit(target)})", capsize=5, marker='o', linestyle='-')
        if plot_any_mean:
            plt.title("Model Predictions by Economic State")
            plt.xlabel("Economic State");
            plt.ylabel("Predicted Value / Mode")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, linestyle=':', alpha=0.7);
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            plt.savefig("plots/prediction_by_economic_state.png", dpi=300)
            logger.info("Plot of predictions (mean/mode) by economic state saved.")
        else:
            logger.warning("No valid data to plot prediction mean/mode by economic state.")
        plt.close()

        # Plot 2
        plt.figure(figsize=(14, 7))
        plot_any_error = False
        for target in self.target_names:
            errors, plot_states_error = [], []
            for state_label_plot_err in self.state_labels:
                target_data_err = state_impact_data_to_plot.get(state_label_plot_err, {}).get(target, {})
                current_error = target_data_err.get('error')
                if current_error is not None and not np.isnan(current_error):
                    errors.append(current_error);
                    plot_states_error.append(state_label_plot_err);
                    plot_any_error = True
            if errors:
                plt.plot(plot_states_error, errors, label=f"{target} ({self._get_target_unit(target)})", marker='o',
                         linestyle='-')
        if plot_any_error:
            error_metric_desc = "Error (Scaled MSE for Regr., 1-Acc for RMESR)"
            plt.title(f"Prediction Error by Economic State");
            plt.xlabel("Economic State");
            plt.ylabel(error_metric_desc)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, linestyle=':', alpha=0.7);
            plt.yscale('log');
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            plt.savefig(f"plots/prediction_error_by_economic_state.png", dpi=300)
            logger.info(f"Prediction error by economic state plot saved.")
        else:
            logger.warning("No valid data to plot prediction error by economic state.")
        plt.close()

    def generate_full_report(self):
        report_content = []
        try:
            if not self.metrics or not self.metrics.get('train') or not self.metrics.get('val'):
                logger.info("Metrics not calculated or incomplete, attempting to calculate now...")
                self.calculate_all_metrics()

            report_content.append("\n===== Enhanced Experiment Results Report V4 (with Test Set) =====")
            report_content.append("\n1. Training Process Summary:")
            train_hist_loss = self.history.get('train_loss', []) if self.history else []
            val_hist_loss = self.history.get('val_loss', []) if self.history else []

            if train_hist_loss and val_hist_loss:
                report_content.append(f"- Training Epochs: {len(train_hist_loss)}")
                min_val_loss_val = min(val_hist_loss) if val_hist_loss and not all(
                    np.isnan(v) for v in val_hist_loss) else float('nan')
                report_content.append(f"- Best Validation Loss: {min_val_loss_val:.4f}")
                report_content.append(f"- Final Training Loss: {train_hist_loss[-1]:.4f}" if train_hist_loss else "N/A")
                report_content.append(f"- Final Validation Loss: {val_hist_loss[-1]:.4f}" if val_hist_loss else "N/A")
            else:
                report_content.append("- Training history is incomplete or not provided.")

            report_content.append("\n2. Model Performance Analysis (Final model performance on various datasets):")
            train_metrics_data = self.metrics.get('train', {})
            val_metrics_data = self.metrics.get('val', {})
            test_metrics_data = self.metrics.get('test', {})

            def format_metric_val(metric_dict, key, fmt="{:.4f}", default_val="N/A"):
                val = metric_dict.get(key)
                return fmt.format(val) if val is not None and not (
                        isinstance(val, float) and np.isnan(val)) else default_val

            for target in self.target_names:
                report_content.append(f"\n- {target} Prediction:")
                train_target_metrics = train_metrics_data.get(target, {})
                val_target_metrics = val_metrics_data.get(target, {})
                test_target_metrics = test_metrics_data.get(target, {})

                is_rmesr_target_report = (target == 'RMESR' and self.num_rmesr_categories > 0)

                if is_rmesr_target_report:
                    report_content.append(
                        f"  Train Set - Acc: {format_metric_val(train_target_metrics, 'accuracy', '{:.2%}')}, "
                        f"F1 (macro): {format_metric_val(train_target_metrics, 'f1')}, "
                        f"ECE: {format_metric_val(train_target_metrics, 'ece', '{:.3f}')}")
                    report_content.append(
                        f"  Validation Set - Acc: {format_metric_val(val_target_metrics, 'accuracy', '{:.2%}')}, "
                        f"F1 (macro): {format_metric_val(val_target_metrics, 'f1')}, "
                        f"ECE: {format_metric_val(val_target_metrics, 'ece', '{:.3f}')}")
                    report_content.append(
                        f"  Test Set - Acc: {format_metric_val(test_target_metrics, 'accuracy', '{:.2%}')}, "
                        f"F1 (macro): {format_metric_val(test_target_metrics, 'f1')}, "
                        f"ECE: {format_metric_val(test_target_metrics, 'ece', '{:.3f}')}")
                else:
                    def get_rmse(metrics_dict):
                        mse = metrics_dict.get('mse')
                        return np.sqrt(mse) if mse is not None and not np.isnan(mse) and mse >= 0 else float('nan')

                    train_rmse = get_rmse(train_target_metrics)
                    val_rmse = get_rmse(val_target_metrics)
                    test_rmse = get_rmse(test_target_metrics)

                    report_content.append(f"  Train Set - RMSE: {train_rmse:.2f}, "
                                          f"MAE: {format_metric_val(train_target_metrics, 'mae', '{:.2f}')}, "
                                          f"R²: {format_metric_val(train_target_metrics, 'r2')}, "
                                          f"sMAPE: {format_metric_val(train_target_metrics, 'smape', '{:.2f}%')}")
                    report_content.append(f"  Validation Set - RMSE: {val_rmse:.2f}, "
                                          f"MAE: {format_metric_val(val_target_metrics, 'mae', '{:.2f}')}, "
                                          f"R²: {format_metric_val(val_target_metrics, 'r2')}, "
                                          f"sMAPE: {format_metric_val(val_target_metrics, 'smape', '{:.2f}%')}")
                    report_content.append(f"  Test Set - RMSE: {test_rmse:.2f}, " 
                                          f"MAE: {format_metric_val(test_target_metrics, 'mae', '{:.2f}')}, "
                                          f"R²: {format_metric_val(test_target_metrics, 'r2')}, "
                                          f"sMAPE: {format_metric_val(test_target_metrics, 'smape', '{:.2f}%')}")
                    report_content.append(
                        f"  Validation Set (Bin) - Acc: {format_metric_val(val_target_metrics, 'accuracy_bin', '{:.2%}')}, "
                        f"F1: {format_metric_val(val_target_metrics, 'f1_bin')}, "
                        f"Thresh: {format_metric_val(val_target_metrics, 'threshold', '{:.2f}')}")

            report_content.append("\n3. Economic State Impact Analysis (Based on Validation Set):")
            state_impact_data = self.metrics.get('state_impact', {})

            avg_errors_by_state = {}
            if state_impact_data:
                for state_label, impact_dict_for_state in state_impact_data.items():
                    if isinstance(impact_dict_for_state, dict) and impact_dict_for_state:
                        valid_errors = [
                            target_data.get('error', float('nan'))
                            for target_data in impact_dict_for_state.values()
                            if isinstance(target_data, dict) and not np.isnan(target_data.get('error', float('nan')))
                        ]
                        avg_errors_by_state[state_label] = np.mean(valid_errors) if valid_errors else float('nan')
                    else:
                        avg_errors_by_state[state_label] = float('nan')
                if any(not np.isnan(v) for v in avg_errors_by_state.values()):
                    sorted_states_by_error = sorted(avg_errors_by_state.items(),
                                                    key=lambda item: item[1] if not np.isnan(item[1]) else float('inf'))
                else:
                    sorted_states_by_error = list(avg_errors_by_state.items())

                for state_l, avg_err_val in sorted_states_by_error:
                    report_content.append(f"\n- {state_l} State (Average Error Metric={avg_err_val:.3f}):")
                    state_specific_impact = state_impact_data.get(state_l, {})
                    for target_n_report in self.target_names:
                        target_stats = state_specific_impact.get(target_n_report, {})
                        unit_str = self._get_target_unit(target_n_report)
                        if target_n_report == 'RMESR' and self.num_rmesr_categories > 0:
                            pred_mode_str = format_metric_val(target_stats, 'pred_mode', "{:.1f}")
                            target_mean_mode_str = format_metric_val(target_stats, 'target_mean_or_mode', "{:.1f}")
                            error_rmesr_str = format_metric_val(target_stats, 'error', "{:.3f}")
                            report_content.append(
                                f"    {target_n_report:<12}: Predicted Mode={pred_mode_str}{unit_str}, Actual Mode/Mean={target_mean_mode_str}{unit_str}, Classification Error={error_rmesr_str}")
                        else:
                            pred_mean_str = format_metric_val(target_stats, 'pred_mean', "{:.1f}")
                            pred_std_str = format_metric_val(target_stats, 'pred_std', "{:.1f}")
                            target_mean_str = format_metric_val(target_stats, 'target_mean', "{:.1f}")
                            error_regr_str = format_metric_val(target_stats, 'error', "{:.2f}")
                            report_content.append(
                                f"    {target_n_report:<12}: Prediction={pred_mean_str}±{pred_std_str}{unit_str}, Actual Mean={target_mean_str}{unit_str}, Error(scaled MSE)={error_regr_str}")
            else:
                report_content.append("- No valid economic state analysis data.")

            report_content.append("\n4. Key Findings and Recommendations:")
            if avg_errors_by_state and not all(np.isnan(v) for v in avg_errors_by_state.values()):
                valid_avg_errors = {k: v for k, v in avg_errors_by_state.items() if not np.isnan(v)}
                if valid_avg_errors:
                    best_state_info = min(valid_avg_errors.items(), key=lambda x: x[1])
                    worst_state_info = max(valid_avg_errors.items(), key=lambda x: x[1])
                    report_content.append(
                        f"- Best Prediction State: {best_state_info[0]} (Average Error Metric={best_state_info[1]:.3f})")
                    report_content.append(
                        f"- Most Challenging State: {worst_state_info[0]} (Average Error Metric={worst_state_info[1]:.3f})")
                    if worst_state_info[0].lower() == 'growth' or 'growth' in worst_state_info[0].lower():
                        report_content.append("  - Recommendation: For the 'growth' state, consider data augmentation, adjusting loss weights, or specific model components.")
                else:
                    report_content.append("- Average error for all states is NaN.")
            else:
                report_content.append("- Failed to calculate effective average state errors.")

            report_content.append("\n5. Feature Importance Analysis (Based on Training Set Subset):")
            if self.feature_importance is not None and not self.feature_importance.empty:
                report_content.append("- Top 10 Most Important Features (Based on Permutation Importance):")
                for _, row in self.feature_importance.head(10).iterrows():
                    report_content.append(f"  {row['feature']:<20}: {row['importance']:.4f}")
            else:
                report_content.append("- Feature importance data was not generated or is empty.")

            report_content.append("\n6. Visualization Plots:")
            plot_paths = []
            if os.path.exists("plots/training_loss_graph_attn.png"): plot_paths.append(
                "plots/training_loss_graph_attn.png")
            if os.path.exists("plots/prediction_by_economic_state.png"): plot_paths.append(
                "plots/prediction_by_economic_state.png")
            if os.path.exists("plots/prediction_error_by_economic_state.png"): plot_paths.append(
                "plots/prediction_error_by_economic_state.png")
            if os.path.exists("plots/feature_importance.png"): plot_paths.append("plots/feature_importance.png")

            if plot_paths:
                for i, p_path in enumerate(plot_paths, 1): report_content.append(f"- Plot {i}: {p_path}")
            else:
                report_content.append("- No generated plots found.")

        except Exception as e_report:
            report_content.append(f"\nA critical error occurred during report generation: {str(e_report)}")
            logger.critical(f"Failed to generate full report: {e_report}", exc_info=True)
        return "\n".join(report_content)