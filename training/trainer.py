import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .losses import DynamicWeightedMultiTaskLoss
logger = logging.getLogger(__name__)


class SIPPModelTrainer:
    """Model training class"""

    def __init__(self, model, device=None, stable_mode=True, config=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.stable_mode = stable_mode  # True for CosineAnnealingLR, False for ReduceLROnPlateau
        self.config = config or {}  # Ensure config is a dictionary

        # Safely get parameters from config, providing default values
        default_lr = 1e-4
        default_wd = 1e-4

        model_output_sizes = []
        if hasattr(model, 'output_sizes') and model.output_sizes:
            model_output_sizes = model.output_sizes
        else:
            logger.warning("SIPPModelTrainer: model.output_sizes is undefined or empty. Will use default single-task size [1].")
            model_output_sizes = [1]

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', default_lr),
            weight_decay=self.config.get('weight_decay', default_wd)
        )
        self.scheduler = None

        self.criterion = None
        if self.config.get('use_dynamic_multitask_loss', False):
            try:
                self.criterion = DynamicWeightedMultiTaskLoss(num_tasks=len(model_output_sizes)).to(self.device)
                logger.info("SIPPModelTrainer: Initialized with DynamicWeightedMultiTaskLoss.")
            except ImportError:
                logger.error("SIPPModelTrainer: DynamicWeightedMultiTaskLoss not found, falling back to fixed weights.")
                self.config['use_dynamic_multitask_loss'] = False
            except Exception as e_dyn_loss:
                logger.error(
                    f"SIPPModelTrainer: Error initializing DynamicWeightedMultiTaskLoss ({e_dyn_loss}). Falling back.")
                self.config['use_dynamic_multitask_loss'] = False

        self.best_val_loss = float('inf')
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        self.early_stopping_counter = 0

    def configure_scheduler(self, train_loader, epochs):
        if self.scheduler is not None:
            logger.info("Scheduler already configured.")
            return

        total_steps = len(train_loader) * epochs if train_loader and epochs else epochs

        if all(k in self.config for k in ['max_lr', 'pct_start', 'div_factor', 'final_div_factor']) and total_steps > 0:
            logger.info("Configuring OneCycleLR scheduler.")
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['max_lr'],
                steps_per_epoch=len(train_loader) if train_loader else 1,
                epochs=epochs,
                pct_start=self.config['pct_start'],
                div_factor=self.config['div_factor'],
                final_div_factor=self.config['final_div_factor']
            )
        elif self.stable_mode and epochs > 0:
            logger.info("Configuring CosineAnnealingLR scheduler (stable_mode=True).")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.config.get('learning_rate', 1e-4) / 100
            )
        elif not self.stable_mode:
            logger.info("Configuring ReduceLROnPlateau scheduler (stable_mode=False).")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
            )
        else:
            logger.warning("Scheduler not configured due to missing parameters or zero epochs/total_steps.")

    def calculate_loss(self, preds, targets, state_logits, economic_state):
        task_losses_raw = []
        current_pred_idx = 0
        target_names = getattr(self.model, 'target_names', [])
        num_rmesr_categories = getattr(self.model, 'num_rmesr_categories', 0)
        model_output_sizes = getattr(self.model, 'output_sizes', [])

        if not model_output_sizes and preds is not None and preds.numel() > 0:
            if len(target_names) == 1:
                model_output_sizes = [preds.shape[1]]
            elif len(target_names) > 1:
                logger.warning(
                    "calculate_loss: model.output_sizes empty for multi-task. Loss calculation might be incorrect. "
                    "Attempting to use config or default.")
                model_output_sizes = self.config.get('output_sizes', [1] * len(target_names))
                if len(model_output_sizes) != len(target_names):
                    model_output_sizes = [preds.shape[1] // len(target_names) if len(target_names) > 0 else preds.shape[
                        1]] * len(target_names)

        rmesr_head_output_idx_in_model = -1
        if target_names and num_rmesr_categories > 0 and model_output_sizes:
            try:
                for i, name_in_model in enumerate(target_names):
                    if name_in_model == 'RMESR' and i < len(model_output_sizes) and \
                            model_output_sizes[i] == num_rmesr_categories:
                        rmesr_head_output_idx_in_model = i
                        break
            except ValueError:
                pass

        rmesr_class_weights_tensor = self.config.get('rmesr_class_weights', None)

        for i, out_size_for_head in enumerate(model_output_sizes):
            if preds is None or current_pred_idx + out_size_for_head > preds.shape[1]:
                logger.error(f"calculate_loss: Prediction slicing out of bounds for task {i}. "
                             f"Preds shape: {preds.shape if preds is not None else 'N/A'}, current_idx={current_pred_idx}, out_size={out_size_for_head}. Skipping task.")
                continue
            task_pred = preds[:, current_pred_idx: current_pred_idx + out_size_for_head]

            if targets is None or i >= targets.shape[1]:
                logger.error(
                    f"calculate_loss: Target index {i} out of bounds for targets shape {targets.shape if targets is not None else 'N/A'}. Skipping task.")
                current_pred_idx += out_size_for_head
                continue
            task_target_values = targets[:, i]

            is_current_head_rmesr = (i == rmesr_head_output_idx_in_model and \
                                     num_rmesr_categories > 0 and \
                                     out_size_for_head == num_rmesr_categories)

            loss = None
            if is_current_head_rmesr:
                task_target_long = task_target_values.long()
                current_rmesr_weights = None
                if rmesr_class_weights_tensor is not None and isinstance(rmesr_class_weights_tensor, torch.Tensor):
                    current_rmesr_weights = rmesr_class_weights_tensor.to(task_pred.device)
                loss = F.cross_entropy(task_pred, task_target_long, weight=current_rmesr_weights)
            else:
                pred_for_mse = task_pred.squeeze(-1) if task_pred.ndim > 1 and task_pred.shape[-1] == 1 else task_pred
                if pred_for_mse.shape != task_target_values.shape:
                    logger.warning(
                        f"    Shape mismatch for MSE loss in task {i}. Pred: {pred_for_mse.shape}, Target: {task_target_values.shape}. Trying to reshape target.")
                loss = F.mse_loss(pred_for_mse, task_target_values)

            if loss is not None:
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.error(
                        f"  !!!! NaN/Inf DETECTED IN TASK LOSS for task {i} ({target_names[i] if i < len(target_names) else 'Unknown'}) !!!!")
                    logger.error(
                        f"     Pred stats: mean={task_pred.mean().item():.4g}, std={task_pred.std().item():.4g}, min={task_pred.min().item():.4g}, max={task_pred.max().item():.4g}")
                    logger.error(
                        f"     Target stats: mean={task_target_values.mean().item():.4g}, std={task_target_values.std().item():.4g}, min={task_target_values.min().item():.4g}, max={task_target_values.max().item():.4g}")
                task_losses_raw.append(loss)
            current_pred_idx += out_size_for_head

        dev = self.device
        if preds is not None and preds.numel() > 0:
            dev = preds.device
        elif targets is not None and targets.numel() > 0:
            dev = targets.device
        elif economic_state is not None and economic_state.numel() > 0:
            dev = economic_state.device

        main_task_loss = torch.tensor(0.0, device=dev, requires_grad=True)
        if not task_losses_raw:
            logger.warning("calculate_loss: No raw task losses were computed. Main task loss will be 0.")
        elif self.criterion is not None and isinstance(self.criterion, DynamicWeightedMultiTaskLoss):
            main_task_loss = self.criterion(task_losses_raw)
        else:
            task_loss_weights_config = self.config.get('task_loss_weights', [1.0] * len(task_losses_raw))
            if len(task_loss_weights_config) != len(task_losses_raw):
                logger.error(
                    f"Task loss weights count ({len(task_loss_weights_config)}) mismatch with actual tasks "
                    f"({len(task_losses_raw)}). Using equal weights."
                )
                task_loss_weights_config = [1.0] * len(task_losses_raw)
            rmesr_extra_multiplier = self.config.get('rmesr_loss_multiplier_if_not_dynamic', 1.0)
            temp_main_loss_sum = 0.0
            for current_task_idx_in_loop, single_task_loss_val in enumerate(task_losses_raw):
                weight_for_this_task = task_loss_weights_config[current_task_idx_in_loop]
                is_this_loop_head_rmesr = (current_task_idx_in_loop == rmesr_head_output_idx_in_model and \
                                           num_rmesr_categories > 0 and \
                                           current_task_idx_in_loop < len(model_output_sizes) and \
                                           model_output_sizes[current_task_idx_in_loop] == num_rmesr_categories)
                if is_this_loop_head_rmesr:
                    weight_for_this_task *= rmesr_extra_multiplier
                temp_main_loss_sum += weight_for_this_task * single_task_loss_val
            main_task_loss = temp_main_loss_sum

        state_loss = torch.tensor(0.0, device=dev)
        if state_logits is not None and economic_state is not None and \
                state_logits.numel() > 0 and economic_state.numel() > 0:
            state_target = economic_state.squeeze().long()
            if state_target.ndim == 0: state_target = state_target.unsqueeze(0)
            if state_logits.shape[0] == state_target.shape[0]:
                try:
                    state_loss = F.cross_entropy(state_logits, state_target)
                except Exception as e_state_loss:
                    logger.error(f"Error calculating state_loss: {e_state_loss}. "
                                 f"state_logits shape: {state_logits.shape}, state_target shape: {state_target.shape}")
            else:
                logger.warning(f"Batch size mismatch for state_loss calculation. "
                               f"state_logits: {state_logits.shape[0]}, state_target: {state_target.shape[0]}. Skipping state loss.")

        total_loss = main_task_loss + self.config.get('state_loss_weight', 0.3) * state_loss
        return total_loss, task_losses_raw

    def train(self, train_loader, val_loader, epochs):
        history = {'train_loss': [], 'val_loss': [], 'task_losses': [], 'val_task_losses': []}
        model_output_sizes_len = len(getattr(self.model, 'output_sizes', []))
        if model_output_sizes_len == 0:
            model_output_sizes_len = len(self.config.get('output_sizes', [1]))
            logger.warning(f"train: model.output_sizes empty, using len from config: {model_output_sizes_len}")
        if self.scheduler is None:
            self.configure_scheduler(train_loader, epochs)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            task_losses_epoch_sum = [0.0] * model_output_sizes_len
            if not train_loader or len(train_loader) == 0:
                logger.warning(f"Epoch {epoch + 1}/{epochs}: train_loader is empty. Skipping training loop.")
                history['train_loss'].append(float('nan'))
                history['task_losses'].append([float('nan')] * model_output_sizes_len)
                val_loss_epoch, val_task_losses_epoch = self._validate(val_loader)
                history['val_loss'].append(val_loss_epoch)
                history['val_task_losses'].append(val_task_losses_epoch)
                if self.early_stopping_counter >= self.early_stopping_patience: break
                continue

            for batch_idx, batch_data in enumerate(train_loader):
                x, y, economic_state, graph_nodes_embeddings, graph_nodes_relevance_mask = batch_data
                x, y = x.to(self.device), y.to(self.device)
                economic_state = economic_state.to(self.device)
                graph_nodes_embeddings = graph_nodes_embeddings.to(self.device)
                graph_nodes_relevance_mask = graph_nodes_relevance_mask.to(self.device)
                self.optimizer.zero_grad()
                predictions, state_logits, _, _ = self.model(
                    x,
                    economic_state=economic_state,
                    batch_graph_node_embeddings=graph_nodes_embeddings,
                    batch_graph_nodes_relevance_mask=graph_nodes_relevance_mask,
                    prev_state_lstm=None
                )
                loss, task_losses_list_raw = self.calculate_loss(predictions, y, state_logits, economic_state)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"Epoch {epoch + 1}, Batch {batch_idx}: NaN/Inf loss. Skipping update. Preds mean: {predictions.mean().item() if predictions is not None else 'N/A'}")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
                self.optimizer.step()
                epoch_loss += loss.item()
                for i, tl in enumerate(task_losses_list_raw):
                    if i < len(task_losses_epoch_sum): task_losses_epoch_sum[i] += tl.item()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                if batch_idx % (max(1, len(train_loader) // 10)) == 0:
                    task_loss_str = [f"{tl.item():.4f}" for tl in task_losses_list_raw]
                    log_msg = (
                        f"E{epoch + 1} B{batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | TasksRaw: {task_loss_str}")
                    if self.criterion and isinstance(self.criterion, DynamicWeightedMultiTaskLoss) and hasattr(
                            self.criterion, 'log_vars'):
                        lvars = self.criterion.log_vars.data.cpu().numpy()
                        prec = np.exp(-lvars)
                        log_msg += f" | DynL lvars:[{','.join(f'{v:.2f}' for v in lvars)}] prec:[{','.join(f'{p:.2f}' for p in prec)}] "
                    logger.info(log_msg)

            avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss
            avg_task_losses_train = [s / len(train_loader) if len(train_loader) > 0 else s for s in
                                     task_losses_epoch_sum]
            history['train_loss'].append(avg_train_loss)
            history['task_losses'].append(avg_task_losses_train)
            val_loss_epoch, val_task_losses_epoch = self._validate(val_loader)
            history['val_loss'].append(val_loss_epoch)
            history['val_task_losses'].append(val_task_losses_epoch)
            lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
            logger.info(
                f"E{epoch + 1} END | TrainL: {avg_train_loss:.4f} | ValL: {val_loss_epoch:.4f} | LR: {lr:.2e}"
            )
            logger.info(f"  AvgRawTrainTasks: {[f'{tl:.4f}' for tl in avg_task_losses_train]}")
            logger.info(f"  AvgRawValTasks  : {[f'{tl:.4f}' for tl in val_task_losses_epoch]}")
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR) and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss_epoch if not np.isnan(val_loss_epoch) else float('inf'))
                else:
                    self.scheduler.step()
            if not np.isnan(val_loss_epoch) and val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
                self.early_stopping_counter = 0
                os.makedirs("results", exist_ok=True)
                try:
                    torch.save(self.model.state_dict(), "results/best_model_graph_attn.pth")
                    logger.info(f"New best val_loss: {self.best_val_loss:.4f}. Model saved.")
                except Exception as e:
                    logger.error(f"Saving best model failed: {e}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}.")
                    break
        return history

    def _validate(self, loader):
        return self._evaluate_epoch(loader)

    def test(self, test_loader):
        """Evaluate the model on the test set"""
        logger.info("--- Starting final evaluation on the test set ---")
        test_loss, test_task_losses = self._evaluate_epoch(test_loader)
        logger.info(f"Test set evaluation complete | Test Loss: {test_loss:.4f}")
        logger.info(f"  AvgRawTestTasks: {[f'{tl:.4f}' for tl in test_task_losses]}")
        return {'test_loss': test_loss, 'test_task_losses': test_task_losses}

    ## MODIFIED/NEW ## - Extract core logic of _validate into a general evaluation function
    def _evaluate_epoch(self, loader):
        """Evaluate the model for one epoch on a given data loader (for validation and testing)"""
        self.model.eval()
        total_loss = 0.0
        model_output_sizes_len = len(getattr(self.model, 'output_sizes', []))
        if model_output_sizes_len == 0:
            model_output_sizes_len = len(self.config.get('output_sizes', [1]))
            logger.warning(
                f"_evaluate_epoch: model.output_sizes empty, using len from config: {model_output_sizes_len}")

        total_task_losses_raw_sum = [0.0] * model_output_sizes_len

        if not loader or len(loader) == 0:
            logger.warning("_evaluate_epoch: Dataloader is empty.")
            return float('nan'), [float('nan')] * model_output_sizes_len

        with torch.no_grad():
            for batch_data in loader:
                x, y, economic_state, graph_nodes_embeddings, graph_nodes_relevance_mask = batch_data
                x, y = x.to(self.device), y.to(self.device)
                economic_state = economic_state.to(self.device)
                graph_nodes_embeddings = graph_nodes_embeddings.to(self.device)
                graph_nodes_relevance_mask = graph_nodes_relevance_mask.to(self.device)

                predictions, state_logits, _, _ = self.model(
                    x,
                    economic_state=economic_state,
                    batch_graph_node_embeddings=graph_nodes_embeddings,
                    batch_graph_nodes_relevance_mask=graph_nodes_relevance_mask,
                    prev_state_lstm=None
                )
                loss, task_losses_list_raw = self.calculate_loss(predictions, y, state_logits, economic_state)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    for i, tl_raw in enumerate(task_losses_list_raw):
                        if i < len(total_task_losses_raw_sum):
                            if not (torch.isnan(tl_raw) or torch.isinf(tl_raw)):
                                total_task_losses_raw_sum[i] += tl_raw.item()
                            else:
                                logger.warning(f"_evaluate_epoch: NaN/Inf in raw task loss {i}.")
                        else:
                            logger.warning(f"_evaluate_epoch: Raw task loss index {i} OOB for sum array.")
                else:
                    logger.warning(f"_evaluate_epoch: NaN/Inf in total val loss.")

        avg_loss = total_loss / len(loader) if len(loader) > 0 else total_loss
        avg_task_losses_raw_res = [s / len(loader) if len(loader) > 0 else s for s in total_task_losses_raw_sum]
        return avg_loss, avg_task_losses_raw_res


    def _plot_losses(self, history, test_loss=None):
        """Plot training curves (enhanced, supports test loss)"""
        plt.figure(figsize=(12, 6))

        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])

        if not train_losses or not val_losses:
            logger.warning("Cannot plot loss curve: insufficient training or validation history data or mismatched key names.")
            if plt.gcf().get_axes():
                plt.close()
            return

        epochs_ran = len(train_losses)
        if epochs_ran == 0:
            logger.warning("Cannot plot loss curve: no epoch data available.")
            if plt.gcf().get_axes(): plt.close()
            return

        plt.plot(range(epochs_ran), train_losses, label='Train Loss', marker='o', markersize=4, linestyle='-')
        plt.plot(range(epochs_ran), val_losses, label='Validation Loss', marker='s', markersize=4, linestyle='-')

        ## MODIFIED/NEW ## - Plot test loss horizontal line
        if test_loss is not None and not np.isnan(test_loss):
            plt.axhline(y=test_loss, color='g', linestyle='--', linewidth=2,
                        label=f'Test Loss: {test_loss:.4f}')

        val_losses_numeric = [v for v in val_losses if not np.isnan(v)]
        if val_losses_numeric:
            best_epoch_idx_overall = val_losses.index(min(val_losses_numeric))
            plt.axvline(x=best_epoch_idx_overall, color='r', linestyle='--', alpha=0.7,
                        label=f'Best Val Loss Epoch ({best_epoch_idx_overall + 1})')
            min_val_loss_val = val_losses[best_epoch_idx_overall]
            x_text_offset = epochs_ran * 0.05 if epochs_ran > 0 else 0.5
            y_range = (max(val_losses_numeric) - min(val_losses_numeric)) if len(val_losses_numeric) > 1 else 1.0
            y_text_offset = min_val_loss_val + y_range * 0.1 if y_range > 1e-6 else min_val_loss_val + 0.1

            plt.annotate(f'Best Val: {min_val_loss_val:.4f}',
                         xy=(best_epoch_idx_overall, min_val_loss_val),
                         xytext=(best_epoch_idx_overall + x_text_offset, y_text_offset),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         fontsize=9)
        else:
            logger.warning("All validation losses are NaN, cannot annotate the best epoch.")

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training, Validation, and Test Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        try:
            plt.savefig('plots/training_loss_graph_attn.png', dpi=300, bbox_inches='tight')
            logger.info("Training curve saved to plots/training_loss_graph_attn.png")
        except Exception as e:
            logger.error(f"Failed to save training curve plot: {e}")
        plt.close()