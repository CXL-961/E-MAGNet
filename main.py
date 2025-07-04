import os
import json
import sys
import logging
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
# Project-specific imports
from data_handling.preprocessor import SIPPDataPreprocessor
from data_handling.dataset import SIPPTimeSeriesDataset
from utils.data_utils import balance_economic_states_oversampling
from models.economic_lstm import EnhancedEconomicStateAwareLSTM
from training.collate import collate_fn
import training.collate as collate_module
from training.trainer import SIPPModelTrainer
from training.losses import DynamicWeightedMultiTaskLoss
from analysis.results import ResultAnalyzer
from missingness.discovery import GraphStructuralMissingnessDiscovery, AdaptiveContextAnalyzerV2
from missingness.explainer import MissingPatternExplainerV2

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Use %(name)s to distinguish module logs
    handlers=[
        logging.FileHandler("training_v3_graph_attention.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# NetworkX version logging
try:
    import pkg_resources

    nx_version = pkg_resources.get_distribution("networkx").version
    logger.info(f"Using NetworkX version: {nx_version}")
    if float(nx_version.split('.')[0]) >= 3:
        logger.warning("Detected NetworkX 3.0+, part of its API may have changed from earlier versions.")
except Exception as e:
    logger.warning(f"Could not check NetworkX version: {e}")


def main():
    try:
        logger.info("=== Initializing SIPP Prediction System (Graph Attention LSTM) ===")
        CONFIG = {
            "batch_size": 16,
            "epochs": 500,
            "hidden_size": 128,
            "output_sizes": [1, 1, 1, 7],
            "num_rmesr_categories": 7,
            "data_path": "sipp/data/final_merged.feather",
            "target_cols": ['TPTOTINC', 'TFINCPOV', 'TFINCPOVT2', 'RMESR'],
            "sequence_length": 12,
            "forecast_horizon": 1,
            "min_mi": 0.01,
            "learning_rate": 5e-5,
            "max_lr": 1e-4,
            "pct_start": 0.3,
            "div_factor": 25,
            "final_div_factor": 1e4,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "missing_embed_size": 32,
            "state_loss_weight": 0.7,
            "task_loss_weights": [2.0, 3.0, 3.0, 25.0],
            "alpha": 1.0,
            "beta": 0.8,
            "use_context_gate_in_lstm": False,
            "use_dynamic_multitask_loss": True,
            "rmesr_loss_multiplier_if_not_dynamic": 10.0,
            "early_stopping_patience": 30,
            "use_graph_attention": True,  # <--- Ensure this line exists and is set to your desired value
            "graph_attention_hidden_dim": 64,  # <--- Ensure this line exists and is set to your desired value
        }
        logger.info(f"Loaded CONFIG: {json.dumps(CONFIG, indent=2)}")

        collate_module._DEVICE_FOR_COLLATE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {collate_module._DEVICE_FOR_COLLATE}")

        logger.info("\n=== Data Preprocessing ===")
        required_dirs = ["results", "plots", "plots/context", "plots/shap", "debug"]
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)

        preprocessor = SIPPDataPreprocessor(
            CONFIG["data_path"],
            sample_size=None,  # Use all 50 individuals' data
            robust_mode=True
        )
        # Pass GSMD-related parameters to the preprocessor (if the preprocessor creates GSMD internally)
        preprocessor.gsmd_creation_params = {
            'min_mi': CONFIG["min_mi"],
            'max_bins': 10,
            'alpha': CONFIG["alpha"],
            'beta': CONFIG["beta"],
            'max_hops': 10,
            'embedding_size': CONFIG["missing_embed_size"]
        }
        processed_data = preprocessor.prepare_model_data()

        gsmd_instance = preprocessor.gsmd
        raw_gsmd_embeddings_dict = {}
        dynamic_rules = None

        if gsmd_instance and gsmd_instance.missing_graph.number_of_nodes() > 0:
            logger.info(
                f"GSMD instance obtained successfully. Graph: {gsmd_instance.missing_graph.number_of_nodes()} nodes, "
                f"{gsmd_instance.missing_graph.number_of_edges()} edges"
            )
            if hasattr(gsmd_instance, 'missing_embeddings') and isinstance(gsmd_instance.missing_embeddings, dict):
                raw_gsmd_embeddings_dict = gsmd_instance.missing_embeddings
                logger.info(f"GSMD embeddings obtained, number of features: {len(raw_gsmd_embeddings_dict)}")
            else:
                logger.warning("GSMD embeddings not found in gsmd_instance.")
            try:
                explainer = MissingPatternExplainerV2(gsmd_instance)
                dynamic_rules = explainer.generate_dynamic_rules(top_k=10)
                logger.info("Dynamic context-aware rules have been generated.")
                example_node = next((n for n in gsmd_instance.missing_graph.nodes() if n.endswith('_missing')), None)
                if example_node:
                    explainer.visualize_context_heatmap(
                        example_node,
                        save_path=f"plots/context/{example_node.replace(':', '_')}_context_heatmap.png"
                    )
            except Exception as e_expl:
                logger.error(f"Explainer processing failed: {e_expl}", exc_info=True)
        else:
            logger.warning("GSMD object not initialized or graph is empty.")

        logger.info("\n=== Preparing Dataset ===")
        dataset = SIPPTimeSeriesDataset(
            processed_data,
            target_columns=CONFIG["target_cols"],
            sequence_length=CONFIG["sequence_length"],
            forecast_horizon=CONFIG["forecast_horizon"]
        )
        logger.info(f"SIPPTimeSeriesDataset number of features: {len(dataset.get_feature_names() or [])}")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot continue. Please check data preprocessing and SIPPTimeSeriesDataset creation.")

        ## Modified/Added ## - Split into training, validation, and test sets (70/15/15)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        if train_size < 1 or val_size < 1 or test_size < 1:
            logger.warning(
                f"Dataset is too small (total {len(dataset)}) to be effectively split into three parts. The entire dataset will be used for training, and a small portion will be copied for validation and testing.")
            train_set_original = dataset
            # Create a small, non-random copy for the validation/test set
            val_test_indices = list(range(min(max(1, int(0.1 * len(dataset))), len(dataset))))
            val_set = Subset(dataset, val_test_indices)
            test_set = Subset(dataset, val_test_indices)
            if not val_test_indices:
                logger.error("Dataset has only one sample, cannot be split effectively. Validation and test sets will be the same as the training set.")
                val_set = dataset
                test_set = dataset
        else:
            train_set_original, val_set, test_set = random_split(
                dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
            )

        logger.info(
            f"Original training set size: {len(train_set_original)}, Validation set size: {len(val_set)}, Test set size: {len(test_set)}")
        ## Modified/Added ## - End

        train_set_balanced = balance_economic_states_oversampling(train_set_original)
        logger.info(f"Balanced training set size: {len(train_set_balanced)}")

        collate_module._GLOBAL_MISSING_EMBED_SIZE = CONFIG["missing_embed_size"]
        lstm_input_feature_names = dataset.get_feature_names()
        if not lstm_input_feature_names:
            logger.warning("Failed to get the list of LSTM input feature names.")

        collate_module._GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW.clear()
        temp_valid_gsmd_embeddings_for_x = []
        current_gmet_row_idx = 0

        if raw_gsmd_embeddings_dict and lstm_input_feature_names:
            for x_feat_idx, x_feat_name in enumerate(lstm_input_feature_names):
                if x_feat_name in raw_gsmd_embeddings_dict:
                    gsmd_emb_np = raw_gsmd_embeddings_dict[x_feat_name]
                    if hasattr(gsmd_emb_np, 'shape') and len(gsmd_emb_np.shape) > 0 and \
                            gsmd_emb_np.shape[0] == CONFIG["missing_embed_size"]:
                        temp_valid_gsmd_embeddings_for_x.append(torch.from_numpy(gsmd_emb_np).float())
                        collate_module._GLOBAL_INPUT_FEATURE_IDX_TO_MISSING_TENSOR_ROW[
                            x_feat_idx] = current_gmet_row_idx
                        current_gmet_row_idx += 1
                    else:
                        logger.warning(f"Feature '{x_feat_name}' GSMD embedding dimension mismatch.")

        if temp_valid_gsmd_embeddings_for_x:
            collate_module._GLOBAL_MISSING_EMB_TENSOR = torch.stack(temp_valid_gsmd_embeddings_for_x).to(
                collate_module._DEVICE_FOR_COLLATE)
            collate_module._NUM_GSMD_MODELLED_NODES = collate_module._GLOBAL_MISSING_EMB_TENSOR.shape[0]  # Set the number of nodes
            logger.info(
                f"Created _GLOBAL_MISSING_EMB_TENSOR for collate_fn (shape: {collate_module._GLOBAL_MISSING_EMB_TENSOR.shape}), "
                f"_NUM_GSMD_MODELLED_NODES: {collate_module._NUM_GSMD_MODELLED_NODES}"
            )
        else:
            collate_module._GLOBAL_MISSING_EMB_TENSOR = torch.zeros(0, CONFIG["missing_embed_size"],
                                                                    device=collate_module._DEVICE_FOR_COLLATE)
            collate_module._NUM_GSMD_MODELLED_NODES = 0
            logger.warning("Failed to create a valid _GLOBAL_MISSING_EMB_TENSOR.")

        logger.info("\n=== Creating Data Loaders ===")
        train_loader = DataLoader(
            train_set_balanced, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn,
            num_workers=0, pin_memory=(collate_module._DEVICE_FOR_COLLATE.type == 'cuda')
        )
        val_loader = DataLoader(
            val_set, batch_size=CONFIG["batch_size"], collate_fn=collate_fn,
            num_workers=0, pin_memory=(collate_module._DEVICE_FOR_COLLATE.type == 'cuda')
        )
        ## Modified/Added ##
        test_loader = DataLoader(
            test_set, batch_size=CONFIG["batch_size"], collate_fn=collate_fn,
            num_workers=0, pin_memory=(collate_module._DEVICE_FOR_COLLATE.type == 'cuda')
        )
        ## Modified/Added ## - End
        if len(train_loader) == 0:
            raise ValueError("Training data loader is empty. Please check the dataset and batch size.")

        logger.info("\n=== Initializing Model ===")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot get sample to determine input_size for the model.")
        sample_x_tensor_from_getitem, _, _ = dataset[0]
        sample_x_features_shape_feat_dim = sample_x_tensor_from_getitem.shape[-1]
        logger.info(f"Model will use input_size = {sample_x_features_shape_feat_dim}")

        model = EnhancedEconomicStateAwareLSTM(
            input_size=sample_x_features_shape_feat_dim,
            hidden_size=CONFIG["hidden_size"],
            output_sizes=CONFIG["output_sizes"],
            num_economic_states=5,
            gsmd_node_embed_size=CONFIG["missing_embed_size"],
            num_rmesr_categories=CONFIG["num_rmesr_categories"],
            target_names=CONFIG["target_cols"],
            use_context_gate=CONFIG["use_context_gate_in_lstm"],
            use_graph_attention=CONFIG["use_graph_attention"],
            graph_attention_hidden_dim=CONFIG["graph_attention_hidden_dim"]
        ).to(collate_module._DEVICE_FOR_COLLATE)
        logger.info(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        logger.info("\n=== Configuring Trainer ===")
        rmesr_class_weights = None
        if 'RMESR' in CONFIG["target_cols"]:
            rmesr_labels_in_train_balanced = []
            rmesr_target_idx_in_cfg = CONFIG["target_cols"].index('RMESR')
            if hasattr(train_set_balanced, '__len__') and len(train_set_balanced) > 0:
                logger.info(f"Extracting RMESR labels from balanced training set (size {len(train_set_balanced)})...")
                for i in range(len(train_set_balanced)):
                    try:
                        _x_sample_bal, y_sample_bal, _state_sample_bal = train_set_balanced[i]
                        if y_sample_bal.ndim == 1 and rmesr_target_idx_in_cfg < len(y_sample_bal):
                            rmesr_labels_in_train_balanced.append(
                                int(round(y_sample_bal[rmesr_target_idx_in_cfg].item())))
                        elif y_sample_bal.ndim == 2 and y_sample_bal.shape[0] == CONFIG["forecast_horizon"] and \
                                rmesr_target_idx_in_cfg < y_sample_bal.shape[1]:
                            rmesr_labels_in_train_balanced.append(
                                int(round(y_sample_bal[0, rmesr_target_idx_in_cfg].item())))
                        else:
                            logger.warning(f"RMESR label extraction: Sample {i}'s target y shape {y_sample_bal.shape} "
                                           f"does not meet expectations (ndim=1 or [forecast_horizon ({CONFIG['forecast_horizon']}), num_targets]) or RMESR index is out of bounds.")
                    except Exception as e:
                        logger.error(f"Failed to extract RMESR label for sample {i}: {e}", exc_info=True)
            if rmesr_labels_in_train_balanced:
                unique_labels, counts = np.unique(rmesr_labels_in_train_balanced, return_counts=True)
                logger.info(f"Balanced training set RMESR label (0-6) distribution: {dict(zip(unique_labels, counts))}")
                if len(unique_labels) > 1:
                    try:
                        class_weights_np = compute_class_weight(
                            class_weight='balanced', classes=unique_labels, y=rmesr_labels_in_train_balanced
                        )
                        rmesr_class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(
                            collate_module._DEVICE_FOR_COLLATE)
                        logger.info(f"RMESR class weights (0-6): {rmesr_class_weights.cpu().numpy()}")
                    except ValueError as e_cw:
                        logger.error(f"Error calculating RMESR class weights: {e_cw}. Unique labels: {unique_labels}")
                else:
                    logger.warning("RMESR has only one class, not using class weights.")
            else:
                logger.warning("Failed to extract RMESR labels to calculate weights.")

        trainer_actual_config = {k: CONFIG[k] for k in [
            'learning_rate', 'weight_decay', 'gradient_clip', 'state_loss_weight',
            'task_loss_weights', 'use_dynamic_multitask_loss',
            'rmesr_loss_multiplier_if_not_dynamic', 'early_stopping_patience',
            'max_lr', 'pct_start', 'div_factor', 'final_div_factor'
        ] if k in CONFIG}
        trainer_actual_config['rmesr_class_weights'] = rmesr_class_weights

        trainer = SIPPModelTrainer(
            model, device=collate_module._DEVICE_FOR_COLLATE, stable_mode=True, config=trainer_actual_config
        )

        logger.info("\n=== Starting Training ===")
        history = trainer.train(train_loader, val_loader, CONFIG["epochs"])

        logger.info("\n=== Evaluating Best Model on Test Set ===")  ## Modified/Added ##
        best_model_path = "results/best_model_graph_attn.pth"
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model {best_model_path} for test evaluation.")
            model.load_state_dict(torch.load(best_model_path, map_location=collate_module._DEVICE_FOR_COLLATE))
        else:
            logger.warning(f"Best model {best_model_path} not found. Will use the final model at the end of training for testing.")

        test_results = trainer.test(test_loader)
        test_loss_val = test_results.get('test_loss', float('nan'))

        logger.info("\n=== Saving Results ===")
        model_path = "results/final_model_graph_attn.pth"
        torch.save(model.state_dict(), model_path)
        config_path = "results/config_graph_attn.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2, ensure_ascii=False)
        if history and ('train_loss' in history or 'train' in history):
            trainer._plot_losses(history, test_loss=test_loss_val)  # ## Modified/Added ## - pass test_loss
        if dynamic_rules is not None and not dynamic_rules.empty:
            dynamic_rules.to_csv("results/dynamic_missing_rules_graph_attn.csv", index=False)

        logger.info(f"Model: {model_path}, Config: {config_path}")

        logger.info("\n=== Initializing Result Analyzer ===")
        analyzer = ResultAnalyzer(  ## Modified/Added ##
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            target_names=CONFIG['target_cols'], device=collate_module._DEVICE_FOR_COLLATE
        )
        if history: analyzer.set_training_history(history)

        if hasattr(train_set_original, '__len__') and len(train_set_original) > 0:
            analyzer.analyze_feature_importance(train_set_original)
        else:
            logger.warning("Original training set is empty, skipping feature importance analysis.")

        report = analyzer.generate_full_report()
        report_path = "results/enhanced_report_v3_graph_attn.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        metrics_data = analyzer.metrics
        metrics_path = "results/metrics_v3_graph_attn.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Report: {report_path}, Metrics: {metrics_path}")
        logger.info("Training and testing completed!")

    except ValueError as ve:
        logger.critical(f"Critical value error led to program termination: {str(ve)}", exc_info=True)
        sys.exit(1)
    except Exception as e_main:
        logger.critical(f"System main process failed to run: {str(e_main)}", exc_info=True)
        if 'CONFIG' in locals() or 'CONFIG' in globals():
            try:
                os.makedirs("results", exist_ok=True)
                with open("results/failed_run_config.json", "w", encoding="utf-8") as f_cfg:
                    json.dump(CONFIG, f_cfg, indent=2, ensure_ascii=False)
                logger.info("An error occurred, attempting to save the current CONFIG to failed_run_config.json")
            except Exception as e_cfg_save:
                logger.error(f"Failed to save CONFIG during error handling: {e_cfg_save}")
        sys.exit(1)
    finally:
        logger.info("=== Program execution finished ===")


if __name__ == "__main__":
    main()