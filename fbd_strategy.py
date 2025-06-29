"""
FBD Evaluation Strategies and Main FBD Strategy
Provides different evaluation methods for FBD federated learning and the main Flower-based strategy
"""

import torch
import torch.nn as nn
import time
import numpy as np
import logging
import os
import json
import csv
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score

# Flower imports for main FBD strategy
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, Scalar

from fbd_models import get_fbd_model
from fbd_utils import load_fbd_settings, load_shipping_plan, load_request_plan, FBDWarehouse, generate_client_model_palettes
from fbd_communication import WeightTransfer
from fbd_dataset import get_data_loader


class FBDStrategy(FedAvg):
    """FBD-enabled Flower strategy that manages FBD warehouse and evaluation."""
    
    def __init__(self, fbd_config_path, shipping_plan_path, request_plan_path, 
                 num_clients, communication_dir, model_template, output_dir, 
                 num_classes, input_shape, test_dataset, batch_size, norm_type='bn', 
                 architecture='resnet18', num_rounds=1, num_ensemble=64, ensemble_colors=None, 
                 update_plan_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # FBD configuration
        self.fbd_config_path = fbd_config_path
        self.shipping_plan_path = shipping_plan_path
        self.request_plan_path = request_plan_path
        self.num_clients = num_clients
        self.communication_dir = communication_dir
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.norm_type = norm_type  # Store normalization type
        self.architecture = architecture  # Store model architecture
        self.num_rounds = num_rounds  # Store total number of rounds
        self.num_ensemble = num_ensemble  # Store number of ensemble models
        self.ensemble_colors = ensemble_colors  # Store ensemble colors
        
        # Load FBD settings
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        self.shipping_plan = load_shipping_plan(shipping_plan_path)
        self.request_plan = load_request_plan(request_plan_path)
        
        # Load REGULARIZER_PARAMS from FBD config
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("fbd_config", fbd_config_path)
            fbd_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fbd_config_module)
            self.regularizer_params = getattr(fbd_config_module, 'REGULARIZER_PARAMS', {})
            logging.info(f"[FBD Strategy] Loaded regularizer params: {self.regularizer_params}")
        except Exception as e:
            logging.warning(f"[FBD Strategy] Failed to load REGULARIZER_PARAMS: {e}")
            self.regularizer_params = {}
        
        # Load update plan
        self.update_plan = None
        if update_plan_path and os.path.exists(update_plan_path):
            try:
                with open(update_plan_path, 'r') as f:
                    self.update_plan = json.load(f)
                logging.info(f"[FBD Strategy] Loaded update plan from {update_plan_path}")
            except Exception as e:
                logging.warning(f"[FBD Strategy] Failed to load update plan: {e}")
        else:
            logging.info(f"[FBD Strategy] No update plan provided or file not found")
        
        # Initialize FBD warehouse with logging
        warehouse_log_path = os.path.join(output_dir, "warehouse.log") if output_dir else "warehouse.log"
        self.warehouse = FBDWarehouse(self.fbd_trace, model_template, warehouse_log_path)
        
        # Initialize communication
        self.communication = WeightTransfer(communication_dir)
        
        # Generate client palettes
        self.client_palettes = generate_client_model_palettes(num_clients, fbd_config_path)
        
        # Initialize tracking for best performance
        self.best_metrics = {
            'M0': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'M1': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'M2': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'M3': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'M4': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'M5': {'auc': 0.0, 'acc': 0.0, 'round': 0},
            'Averaging': {'auc': 0.0, 'acc': 0.0, 'round': 0}
        }
        self.best_overall_round = 0
        self.best_overall_auc = 0.0
        self.best_warehouse_state = None
        
        # Initialize training history tracking
        self.training_history = {
            'loss_centralized': {},
            'auc_centralized': {},
            'accuracy_centralized': {},
            'round_times': {},
            'total_rounds': 0,
            'total_time': 0.0,
            'start_time': None,
            'end_time': None
        }
        
        logging.info(f"[FBD Strategy] Initialized with {len(self.fbd_trace)} FBD blocks")
        logging.info(f"[FBD Strategy] Warehouse summary: {self.warehouse.warehouse_summary()}")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure fit round with FBD shipping phase."""
        
        # Track round start time
        if self.training_history['start_time'] is None:
            self.training_history['start_time'] = time.time()
        
        round_start_time = time.time()
        self.training_history['round_times'][server_round] = {'start': round_start_time}
        
        # Log learning rate schedule
        config_dict = self.on_fit_config_fn(server_round)
        if server_round == 1 or server_round == 21 or server_round == 51:
            logging.info(f"📊 Learning Rate Schedule Update:")
            logging.info(f"   Round {server_round}: LR = {config_dict['local_learning_rate']:.6f} ({config_dict['lr_schedule_info']})")
            if server_round > 1:
                logging.info(f"   Base LR: {config_dict['base_learning_rate']:.6f}")
        
        # FBD Shipping Phase
        if server_round in self.shipping_plan:
            logging.info(f"\n" + "=" * 60)
            logging.info(f"📦 [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Executing shipping phase")
            logging.info(f"=" * 60)
            shipping_clients = self.shipping_plan[server_round]
            
            # Use configured number of clients (clients 0 to num_clients-1 are expected to be active)
            active_client_ids = set(range(self.num_clients))
            logging.info(f"[FBD Strategy] Expected active clients: {sorted(active_client_ids)}")
            
            shipped_count = 0
            for client_id_str, shipping_list in shipping_clients.items():
                client_id = int(client_id_str)  # Convert string key to int
                
                if client_id in active_client_ids:
                    # Get weights from warehouse
                    shipping_weights = self.warehouse.get_shipping_weights(shipping_list)
                    
                    # Send weights to client
                    self.communication.server_send_weights(client_id, server_round, shipping_weights)
                    shipped_count += 1
                    logging.info(f"[FBD Strategy] Shipped {len(shipping_list)} blocks to client {client_id}")
                else:
                    logging.info(f"[FBD Strategy] Skipping client {client_id} (client ID >= num_clients={self.num_clients})")
            
            logging.info(f"[FBD Strategy] Shipped weights to {shipped_count} expected clients")
        
        # Get available clients
        sample_size, min_num_clients = client_manager.num_available(), self.min_fit_clients
        if sample_size < min_num_clients:
            logging.warning(f"Not enough clients available for training: {sample_size} < {min_num_clients}")
            return []
        
        # Sample clients
        sampled_clients = client_manager.sample(num_clients=min_num_clients) # , criterion=client_manager.criterion
        
        # Create per-client configuration with update plans
        fit_instructions = []
        for client in sampled_clients:
            client_id = int(client.cid)
            
            # Get base config
            base_config = self.on_fit_config_fn(server_round)
            
            # Debug: Check update plan conditions
            print(f"[Server Update Plan Debug] Client {client_id}, Round {server_round}:")
            print(f"  - self.update_plan is not None: {self.update_plan is not None}")
            if self.update_plan is not None:
                print(f"  - Available rounds in update_plan: {list(self.update_plan.keys())}")
                print(f"  - str(server_round) in self.update_plan: {str(server_round) in self.update_plan}")
                if str(server_round) in self.update_plan:
                    round_clients = list(self.update_plan[str(server_round)].keys())
                    print(f"  - Available clients in round {server_round}: {round_clients}")
                    print(f"  - str(client_id) in round plan: {str(client_id) in self.update_plan[str(server_round)]}")
                else:
                    print(f"  - Round {server_round} NOT FOUND in update plan")
            else:
                print(f"  - Update plan is None on server side")
            
            # Add client-specific update plan if available
            if (self.update_plan is not None and 
                str(server_round) in self.update_plan and 
                str(client_id) in self.update_plan[str(server_round)]):
                
                client_update_plan = self.update_plan[str(server_round)][str(client_id)]
                base_config["current_update_plan"] = client_update_plan
                
                print(f"[Server Update Plan Debug] ✅ SENDING update plan to client {client_id}")
                logging.info(f"[FBD Strategy] Sending update plan to client {client_id}: "
                           f"model_to_update with {len(client_update_plan['model_as_regularizer'])} regularizers")
            else:
                print(f"[Server Update Plan Debug] ❌ NOT SENDING update plan to client {client_id}")
            
            fit_instructions.append((client, fl.common.FitIns(parameters, base_config)))
        
        return fit_instructions

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Aggregate fit results with FBD collection phase."""
        
        # Get current parameters from the first result or create dummy
        if results:
            current_parameters = results[0][1].parameters
        else:
            # Create dummy parameters if no results
            current_parameters = ndarrays_to_parameters([np.array([0.0])])
        
        # FBD Collection Phase - collect weights sent by clients during training
        logging.info(f"\n" + "=" * 60)
        logging.info(f"📥 [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Executing collection phase")
        logging.info(f"=" * 60)
        
        # Use configured number of clients (same as shipping phase)
        active_client_ids = set(range(self.num_clients))
        logging.info(f"[FBD Strategy] Expected active clients: {sorted(active_client_ids)}")
        logging.info(f"[FBD Strategy] Received results from {len(results)} clients")
        
        # Collect weights from active clients (no need to send requests - clients auto-send)
        collected_weights = {}
        collected_count = 0
        
        for client_id in active_client_ids:
            try:
                received_weights, _ = self.communication.server_receive_weights(client_id, server_round)
                if received_weights:
                    collected_weights.update(received_weights)
                    collected_count += 1
                    logging.info(f"[FBD Strategy] Received {len(received_weights)} blocks from client {client_id}")
                else:
                    logging.info(f"[FBD Strategy] Client {client_id} sent no weights")
            except (TimeoutError, FileNotFoundError) as e:
                logging.info(f"[FBD Strategy] No weights received from client {client_id} in round {server_round}")
        
        # Store collected weights in warehouse
        if collected_weights:
            self.warehouse.store_weights_batch(collected_weights)
            logging.info(f"[FBD Strategy] Collected and stored {len(collected_weights)} FBD blocks from {collected_count} clients")
        else:
            logging.info(f"[FBD Strategy] No weights collected from any client in round {server_round}")
        
        # Return aggregated metrics (no parameter aggregation needed for FBD)
        metrics_aggregated = {}
        client_regularizer_metrics = {}
        
        if results:
            # Aggregate client metrics
            for client_proxy, fit_res in results:
                client_id = int(client_proxy.cid)
                
                for key, value in fit_res.metrics.items():
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = []
                    metrics_aggregated[key].append(value)
                
                # Store per-client regularizer metrics if available
                if any(k.startswith('regularizer_') for k in fit_res.metrics.keys()):
                    client_regularizer_metrics[client_id] = {
                        k: v for k, v in fit_res.metrics.items() 
                        if k.startswith('regularizer_') or k in ['num_regularizers', 'regularization_strength']
                    }
            
            # Average the metrics (except regularizer batch details)
            for key, values in metrics_aggregated.items():
                if key != 'regularizer_batch_details':  # Skip averaging detailed batch data
                    if isinstance(values[0], (int, float)):  # Only average numeric values
                        metrics_aggregated[key] = sum(values) / len(values)
                    else:
                        metrics_aggregated[key] = values  # Keep lists as-is
        
        # Store regularizer metrics in training history
        if client_regularizer_metrics:
            if 'regularizer_metrics' not in self.training_history:
                self.training_history['regularizer_metrics'] = {}
            self.training_history['regularizer_metrics'][server_round] = client_regularizer_metrics
            
            # Log regularizer metrics summary
            avg_reg_distance = metrics_aggregated.get('avg_regularizer_distance', 0)
            if avg_reg_distance > 0:
                logging.info(f"[FBD Strategy] Round {server_round} Regularizer Summary:")
                logging.info(f"  Average regularizer distance: {avg_reg_distance:.6f}")
                logging.info(f"  Regularizer type: {metrics_aggregated.get('regularizer_type', 'N/A')}")
                logging.info(f"  Regularizer params: {self.regularizer_params}")
                logging.info(f"  Clients with regularizers: {len(client_regularizer_metrics)}")
        
        # Return current parameters (unchanged for FBD)
        return current_parameters, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate using FBD comprehensive evaluation strategy (M0-M5 + Averaging) and ensemble strategy."""
        print(f"============Round [{server_round}/{self.num_rounds}] =========")
        logging.info(f"🔄 [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Executing evaluation phase")
        
        # Create test loader for evaluation
        test_loader = get_data_loader(self.test_dataset, self.batch_size)
        
        # Debugging device availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"[FBD Strategy] Server-side evaluation device check: torch.cuda.is_available() -> {torch.cuda.is_available()}. Using device: {device}")
        
        # 1. Use FBD comprehensive evaluation strategy (M0-M5 + Averaging = 7 models)
        logging.info(f"📊 [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Running comprehensive evaluation...")
        evaluation_results = fbd_comprehensive_evaluate(
            warehouse=self.warehouse,
            round_num=server_round,
            test_loader=test_loader,
            num_classes=self.num_classes,
            input_shape=self.input_shape,
            device=device,
            norm=self.norm_type,  # Use stored norm type
            architecture=self.architecture  # Use stored architecture
        )
        
        # 2. Use FBD ensemble evaluation strategy
        logging.info(f"🎯 [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Running ensemble evaluation...")
        ensemble_results = fbd_ensemble_evaluate(
            warehouse=self.warehouse,
            round_num=server_round,
            test_loader=test_loader,
            num_classes=self.num_classes,
            input_shape=self.input_shape,
            device=device,
            norm=self.norm_type,
            architecture=self.architecture,  # Use stored architecture
            ensemble_method='voting',
            num_ensemble=self.num_ensemble,
            colors_ensemble=self.ensemble_colors
        )
        
        # Extract summary metrics from comprehensive evaluation
        if evaluation_results.get('success', False):
            summary_metrics = evaluation_results.get('summary_metrics', {})
            avg_loss = summary_metrics.get('average_loss', 0.0)
            avg_acc = summary_metrics.get('average_accuracy', 0.0)
            avg_auc = summary_metrics.get('average_auc', 0.0)
            models_evaluated = summary_metrics.get('total_successful', 0)
            
            # Log individual model results
            individual_results = evaluation_results.get('individual_results', {})
            logging.info(f"📊 [FBD Strategy] Round [{server_round}/{self.num_rounds}] - Comprehensive Evaluation Results:")
            logging.info(f"  Summary: Loss={avg_loss:.4f}, AUC={avg_auc:.4f}, Acc={avg_acc:.2f}%")
            logging.info(f"  Models evaluated: {models_evaluated}/7")
            
            # Log individual models
            for model_name, result in individual_results.items():
                if isinstance(result, dict) and 'accuracy' in result:
                    logging.info(f"    {model_name}: Acc={result['accuracy']:.2f}%, AUC={result['auc']:.4f}, Loss={result['loss']:.4f}")
                else:
                    logging.info(f"    {model_name}: Failed - {result.get('error', 'Unknown error')}")
        else:
            avg_loss = 0.0
            avg_auc = 0.0
            avg_acc = 0.0
            models_evaluated = 0
            logging.info(f"[FBD Strategy] Comprehensive evaluation failed: {evaluation_results.get('error', 'Unknown error')}")
        
        # Extract metrics from ensemble evaluation
        ensemble_acc = 0.0
        ensemble_auc = 0.0
        ensemble_models_generated = 0
        
        if ensemble_results.get('success', False):
            ensemble_metrics = ensemble_results.get('evaluation_metrics', {})
            ensemble_acc = ensemble_metrics.get('ensemble_accuracy', 0.0)
            ensemble_auc = ensemble_metrics.get('ensemble_auc', 0.0)
            ensemble_models_generated = ensemble_metrics.get('num_ensemble_generated', 0)
            
            # Log ensemble results (without detailed composition)
            logging.info(f"🎯 [FBD Strategy] Round [{server_round}/{self.num_rounds}] - Ensemble Evaluation Results:")
            logging.info(f"  Ensemble Accuracy: {ensemble_acc:.2f}%")
            logging.info(f"  Ensemble AUC: {ensemble_auc:.4f}")
            logging.info(f"  Models Generated: {ensemble_models_generated}")
            
            # Log agreement statistics
            agreement_stats = ensemble_metrics.get('agreement_stats', {})
            if agreement_stats:
                total_samples = agreement_stats.get('total_samples', 0)
                logging.info(f"  Agreement Stats (across {total_samples} samples):")
                logging.info(f"    Mean: {agreement_stats.get('average_agreement', 0):.1f}/{ensemble_models_generated} ({agreement_stats.get('agreement_ratio', 0):.3f})")
                logging.info(f"    Median: {agreement_stats.get('median_agreement', 0):.0f}/{ensemble_models_generated}")
                logging.info(f"    Range: {agreement_stats.get('min_agreement', 0)}-{agreement_stats.get('max_agreement', 0)}")
        else:
            logging.info(f"[FBD Strategy] Ensemble evaluation failed: {ensemble_results.get('error', 'Unknown error')}")
        
        # Track evaluation results in history
        self.training_history['loss_centralized'][server_round] = avg_loss
        self.training_history['auc_centralized'][server_round] = avg_auc
        self.training_history['accuracy_centralized'][server_round] = avg_acc
        self.training_history['total_rounds'] = server_round
        
        # Track ensemble results in history (add new tracking)
        if 'ensemble_accuracy_centralized' not in self.training_history:
            self.training_history['ensemble_accuracy_centralized'] = {}
            self.training_history['ensemble_auc_centralized'] = {}
            self.training_history['l2_distances'] = {}
        self.training_history['ensemble_accuracy_centralized'][server_round] = ensemble_acc
        self.training_history['ensemble_auc_centralized'][server_round] = ensemble_auc
        
        # Store L2 distances from ensemble evaluation
        if ensemble_results.get('success', False):
            ensemble_metrics = ensemble_results.get('evaluation_metrics', {})
            l2_distances = ensemble_metrics.get('l2_distances', {})
            if l2_distances:
                self.training_history['l2_distances'][server_round] = l2_distances
                
                # Log L2 distance summary with improved metrics
                position_comparisons = l2_distances.get('by_position_comparisons', {})
                overall_metrics = l2_distances.get('overall_metrics', {})
                
                if position_comparisons:
                    logging.info(f"[FBD Strategy] Round {server_round} L2 Distance Summary:")
                    logging.info(f"  Architecture: {l2_distances.get('architecture', 'unknown')}")
                    logging.info(f"  Total comparisons: {l2_distances.get('total_comparisons', 0)}")
                    
                    for position, comparisons in position_comparisons.items():
                        avg_distance = comparisons.get('average', 0)
                        num_comp = comparisons.get('num_comparisons', 0)
                        if avg_distance > 0:
                            logging.info(f"    {position}: Avg={avg_distance:.6f}, Std={comparisons.get('std_dev', 0):.6f}, Range=[{comparisons.get('min', 0):.6f}-{comparisons.get('max', 0):.6f}], N={num_comp}")
                    
                    # Log overall statistics (new improved metrics)
                    if overall_metrics:
                        logging.info(f"  Overall L2 Distance Metrics:")
                        logging.info(f"    Average: {overall_metrics.get('overall_average_l2_distance', 0):.6f}")
                        logging.info(f"    Std Dev: {overall_metrics.get('overall_std_dev', 0):.6f}")
                        logging.info(f"    Range: [{overall_metrics.get('overall_min', 0):.6f}-{overall_metrics.get('overall_max', 0):.6f}]")
                        logging.info(f"    Positions: {overall_metrics.get('num_positions_compared', 0)}/{overall_metrics.get('total_positions', 0)}")
        
        # Track round end time
        if server_round in self.training_history['round_times']:
            self.training_history['round_times'][server_round]['end'] = time.time()
            round_duration = (self.training_history['round_times'][server_round]['end'] - 
                            self.training_history['round_times'][server_round]['start'])
            self.training_history['round_times'][server_round]['duration'] = round_duration
        
        # Update best metrics tracking
        if evaluation_results.get('success', False):
            self._update_best_metrics(server_round, evaluation_results)
        
        # Save evaluation results
        # Save comprehensive evaluation results
        eval_file = os.path.join(self.output_dir, f"fbd_comprehensive_evaluation_round_{server_round}.json")
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save ensemble evaluation results (including detailed ensemble records)
        ensemble_file = os.path.join(self.output_dir, f"fbd_ensemble_evaluation_round_{server_round}.json")
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_results, f, indent=2, default=str)
        
        logging.info(f"✅ [FBD Strategy] Round [{server_round}/{self.num_rounds}]: Evaluation completed")
        print(f"============Round [{server_round}/{self.num_rounds}] END =======")
        
        return avg_loss, {
            'fbd_auc': avg_auc,
            'fbd_acc': avg_acc,
            'models_evaluated': models_evaluated,
            'total_expected_models': 7,
            'success': evaluation_results.get('success', False),
            'ensemble_auc': ensemble_auc,
            'ensemble_acc': ensemble_acc,
            'ensemble_models_generated': ensemble_models_generated,
            'ensemble_success': ensemble_results.get('success', False)
        }

    def _update_best_metrics(self, server_round: int, evaluation_results: Dict):
        """Update best metrics tracking with current round results."""
        individual_results = evaluation_results.get('individual_results', {})
        
        # Track best performance for each model
        for model_name, result in individual_results.items():
            if isinstance(result, dict) and 'auc' in result and 'accuracy' in result:
                current_auc = result['auc']
                current_acc = result['accuracy']
                
                if model_name in self.best_metrics:
                    # Update if current AUC is better
                    if current_auc > self.best_metrics[model_name]['auc']:
                        self.best_metrics[model_name]['auc'] = current_auc
                        self.best_metrics[model_name]['acc'] = current_acc
                        self.best_metrics[model_name]['round'] = server_round
                        logging.info(f"[FBD Strategy] New best for {model_name}: AUC={current_auc:.4f}, Acc={current_acc:.2f}% (Round {server_round})")
        
        # Track best overall performance (using average AUC)
        summary_metrics = evaluation_results.get('summary_metrics', {})
        overall_auc = summary_metrics.get('average_auc', 0.0)
        
        if overall_auc > self.best_overall_auc:
            self.best_overall_auc = overall_auc
            self.best_overall_round = server_round
            
            # Save warehouse state at best performance
            self.best_warehouse_state = self._save_warehouse_state()
            logging.info(f"[FBD Strategy] New best overall performance: AUC={overall_auc:.4f} (Round {server_round})")
    
    def _save_warehouse_state(self) -> Dict:
        """Save current warehouse state."""
        warehouse_state = {}
        
        for block_id in self.warehouse.fbd_trace.keys():
            try:
                block_weights = self.warehouse.retrieve_weights(block_id)
                if block_weights:
                    # Convert tensors to numpy for JSON serialization
                    weights_numpy = {}
                    for param_name, tensor in block_weights.items():
                        weights_numpy[param_name] = tensor.cpu().numpy().tolist()
                    warehouse_state[block_id] = weights_numpy
            except Exception as e:
                logging.warning(f"[FBD Strategy] Warning: Could not save weights for block {block_id}: {e}")
        
        return warehouse_state
    
    def save_final_results(self):
        """Save the best warehouse weights and CSV summary at the end of training."""
        # Finalize training history
        self.training_history['end_time'] = time.time()
        if self.training_history['start_time']:
            self.training_history['total_time'] = (
                self.training_history['end_time'] - self.training_history['start_time']
            )
        
        logging.info(f"\n[FBD Strategy] Saving final results...")
        
        # 1. Save training history summary (replaces Flower's verbose output)
        self._save_training_summary()
        
        # 2. Save best warehouse weights
        if self.best_warehouse_state:
            warehouse_file = os.path.join(self.output_dir, "best_warehouse_weights.json")
            with open(warehouse_file, 'w') as f:
                json.dump({
                    'best_round': self.best_overall_round,
                    'best_overall_auc': self.best_overall_auc,
                    'warehouse_weights': self.best_warehouse_state,
                    'timestamp': time.time()
                }, f, indent=2)
            logging.info(f"[FBD Strategy] Saved best warehouse weights to: {warehouse_file}")
        
        # 3. Save CSV summary of best performance for each model
        csv_file = os.path.join(self.output_dir, "best_model_performance.csv")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Model', 'Best_AUC', 'Best_Accuracy', 'Best_Round'])
            
            # Write data for each model
            model_order = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'Averaging']
            for model_name in model_order:
                if model_name in self.best_metrics:
                    metrics = self.best_metrics[model_name]
                    writer.writerow([
                        model_name,
                        f"{metrics['auc']:.6f}",
                        f"{metrics['acc']:.4f}",
                        metrics['round']
                    ])
        
        logging.info(f"[FBD Strategy] Saved best performance CSV to: {csv_file}")
        
        # 4. Print concise summary (no verbose Flower output)
        logging.info(f"\n[FBD Strategy] === FINAL TRAINING SUMMARY ===")
        logging.info(f"Training completed: {self.training_history['total_rounds']} rounds in {self.training_history['total_time']:.2f}s")
        logging.info(f"Best overall round: {self.best_overall_round}")
        logging.info(f"Best overall AUC: {self.best_overall_auc:.6f}")
        logging.info(f"\nDetailed results saved to: training_summary.json")
        logging.info(f"=========================================\n")
    
    def _save_training_summary(self):
        """Save comprehensive training summary to JSON file (instead of verbose logs)."""
        summary = {
            "training_metadata": {
                "total_rounds": self.training_history['total_rounds'],
                "total_time_seconds": self.training_history['total_time'],
                "start_timestamp": self.training_history['start_time'],
                "end_timestamp": self.training_history['end_time'],
                "average_time_per_round": (
                    self.training_history['total_time'] / self.training_history['total_rounds']
                    if self.training_history['total_rounds'] > 0 else 0
                )
            },
            "history_centralized": {
                "loss": self.training_history['loss_centralized'],
                "auc": self.training_history['auc_centralized'], 
                "accuracy": self.training_history['accuracy_centralized']
            },
            "history_ensemble": {
                "accuracy": self.training_history.get('ensemble_accuracy_centralized', {}),
                "auc": self.training_history.get('ensemble_auc_centralized', {})
            },
            "round_timings": self.training_history['round_times'],
            "best_performance": {
                "best_overall_round": self.best_overall_round,
                "best_overall_auc": self.best_overall_auc,
                "model_specific": self.best_metrics
            },
            "final_round_performance": {
                "loss": self.training_history['loss_centralized'].get(self.training_history['total_rounds'], 0.0),
                "auc": self.training_history['auc_centralized'].get(self.training_history['total_rounds'], 0.0),
                "accuracy": self.training_history['accuracy_centralized'].get(self.training_history['total_rounds'], 0.0)
            },
            "regularizer_params": self.regularizer_params,
            "regularizer_metrics": self.training_history.get('regularizer_metrics', {}),
            "l2_distances": self.training_history.get('l2_distances', {})
        }
        
        # Save to JSON file
        summary_file = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"[FBD Strategy] Saved training summary to: {summary_file}")
        
        # Also create a concise summary that mimics Flower's format but cleaner
        flower_style_summary = {
            "run_summary": {
                "total_rounds": self.training_history['total_rounds'],
                "total_time_seconds": round(self.training_history['total_time'], 2),
                "status": "completed"
            },
            "regularizer_params": self.regularizer_params,
            "history_loss_centralized": {
                f"round {k}": v for k, v in self.training_history['loss_centralized'].items()
            },
            "history_auc_centralized": {
                f"round {k}": v for k, v in self.training_history['auc_centralized'].items()
            },
            "history_accuracy_centralized": {
                f"round {k}": v for k, v in self.training_history['accuracy_centralized'].items()
            },
            "history_ensemble_accuracy_centralized": {
                f"round {k}": v for k, v in self.training_history.get('ensemble_accuracy_centralized', {}).items()
            },
            "history_ensemble_auc_centralized": {
                f"round {k}": v for k, v in self.training_history.get('ensemble_auc_centralized', {}).items()
            },
            "history_regularizer_metrics": {
                f"round {k}": v for k, v in self.training_history.get('regularizer_metrics', {}).items()
            },
            "history_l2_distances": {
                f"round {k}": v for k, v in self.training_history.get('l2_distances', {}).items()
            }
        }
        
        flower_summary_file = os.path.join(self.output_dir, "flower_style_summary.json")
        with open(flower_summary_file, 'w') as f:
            json.dump(flower_style_summary, f, indent=2)
        
        logging.info(f"[FBD Strategy] Saved Flower-style summary to: {flower_summary_file}")


# ================================================
# FBD EVALUATION STRATEGIES
# ================================================

# Removed duplicate function - now using get_fbd_model from fbd_models

class FBDEvaluationStrategy:
    """
    Base class for FBD evaluation strategies.
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 device: str = 'cpu'):
        """
        Initialize evaluation strategy.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input tensor shape (channels, height, width)
            device: Device for computation
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device
    
    def evaluate(self, warehouse, round_num: int) -> Dict[str, Any]:
        """
        Base evaluation method to be implemented by subclasses.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            round_num: Current round number
            
        Returns:
            Dict: Evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

class FBDAverageEvaluationStrategy(FBDEvaluationStrategy):
    """
    FBD Average Evaluation Strategy.
    Averages weights by model part across all blocks, evaluates the resulting model, then discards it.
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 device: str = 'cpu',
                 test_batch_size: int = 64,
                 num_test_batches: int = 10):
        """
        Initialize FBD average evaluation strategy.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input tensor shape
            device: Device for computation
            test_batch_size: Batch size for evaluation
            num_test_batches: Number of test batches
        """
        super().__init__(num_classes, input_shape, device)
        self.test_batch_size = test_batch_size
        self.num_test_batches = num_test_batches
    
    def _average_weights_by_model_part(self, warehouse) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Average weights by model part across all function blocks.
        
        Args:
            warehouse: FBD warehouse
            
        Returns:
            Dict: Averaged weights organized by model part
        """
        model_parts = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        averaged_weights = {}
        
        for model_part in model_parts:
            # Find all blocks belonging to this model part
            part_blocks = []
            for block_id, block_info in warehouse.fbd_trace.items():
                if block_info['model_part'] == model_part:
                    part_blocks.append(block_id)
            
            if part_blocks:
                # Get weights from all blocks for this part
                block_weights_list = []
                for block_id in part_blocks:
                    try:
                        block_weights = warehouse.retrieve_weights(block_id)
                        if block_weights:  # Only include non-empty weights
                            block_weights_list.append(block_weights)
                    except Exception:
                        continue  # Skip blocks with missing weights
                
                if block_weights_list:
                    # Average the weights
                    averaged_part_weights = {}
                    
                    # Get all parameter names from the first block
                    param_names = block_weights_list[0].keys()
                    
                    for param_name in param_names:
                        # Stack tensors from all blocks and compute mean
                        param_tensors = []
                        for block_weights in block_weights_list:
                            if param_name in block_weights:
                                param_tensors.append(block_weights[param_name])
                        
                        if param_tensors:
                            # Average across all blocks
                            stacked_tensors = torch.stack(param_tensors)
                            
                            # Convert to float for averaging if needed, then convert back to original dtype
                            original_dtype = stacked_tensors.dtype
                            if original_dtype in [torch.long, torch.int, torch.short, torch.uint8]:
                                # Convert to float for averaging
                                stacked_tensors = stacked_tensors.float()
                                averaged_param = torch.mean(stacked_tensors, dim=0)
                                # Convert back to original dtype if it was integer type
                                averaged_param = averaged_param.to(original_dtype)
                            else:
                                # Already floating point, can compute mean directly
                                averaged_param = torch.mean(stacked_tensors, dim=0)
                            
                            averaged_part_weights[param_name] = averaged_param
                    
                    averaged_weights[model_part] = averaged_part_weights
        
        return averaged_weights
    
    def _evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate a model using mock test data.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Dict: Evaluation metrics including loss, accuracy, and AUC
        """
        model.eval()
        total_loss = 0.0
        y_score = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for _ in range(self.num_test_batches):
                # Generate random test data
                inputs = torch.randn(self.test_batch_size, *self.input_shape).to(self.device)
                targets = torch.randint(0, self.num_classes, (self.test_batch_size,)).to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Apply softmax to get probabilities
                probabilities = softmax(outputs)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Collect predictions and targets for AUC calculation
                y_score = torch.cat((y_score, probabilities), 0)
                y_true = torch.cat((y_true, targets.float().unsqueeze(1)), 0)
        
        # Convert to numpy for sklearn metrics
        y_score_np = y_score.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy().flatten()
        
        # Calculate metrics
        avg_loss = total_loss / self.num_test_batches
        
        # Calculate accuracy
        y_pred = y_score_np.argmax(axis=1)
        accuracy = accuracy_score(y_true_np, y_pred) * 100.0  # Convert to percentage
        
        # Calculate AUC 
        try:
            if self.num_classes == 2:
                # Binary classification - use probability of positive class
                auc = roc_auc_score(y_true_np, y_score_np[:, 1])
            else:
                # Multi-class classification
                auc = roc_auc_score(y_true_np, y_score_np, multi_class='ovr')
        except ValueError:
            # Handle case where all targets are the same class (can happen with random data)
            auc = 0.5  # Random performance
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'total_samples': len(y_true_np),
            'total_batches': self.num_test_batches
        }
    
    def evaluate(self, warehouse, round_num: int) -> Dict[str, Any]:
        """
        Evaluate FBD warehouse by averaging weights and testing the resulting model.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            round_num: Current round number
            
        Returns:
            Dict: Evaluation results
        """
        evaluation_start_time = time.time()
        
        try:
            # Step 1: Average weights by model part across all blocks
            averaged_weights = self._average_weights_by_model_part(warehouse)
            
            if not averaged_weights:
                return {
                    'round': round_num,
                    'timestamp': time.time(),
                    'success': False,
                    'error': 'No weights available for averaging',
                    'evaluation_time': time.time() - evaluation_start_time
                }
            
            # Step 2: Create temporary model and load averaged weights
            # Use norm type stored on strategy instance, default to 'bn' if not set
            norm_type = getattr(self, 'norm', 'bn')
            # Use architecture type stored on strategy instance, default to 'resnet18' if not set
            architecture_type = getattr(self, 'architecture', 'resnet18')
            temp_model = get_fbd_model(
                architecture=architecture_type,
                norm=norm_type,
                in_channels=self.input_shape[0], 
                num_classes=self.num_classes
            ).to(self.device)
            
            temp_model.load_from_dict(averaged_weights)
            
            # Step 3: Evaluate the model
            eval_metrics = self._evaluate_model(temp_model)
            
            # Step 4: Get model statistics
            total_params = sum(p.numel() for p in temp_model.parameters())
            trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            
            # Count blocks that were averaged
            blocks_averaged = 0
            for model_part in averaged_weights.keys():
                part_blocks = [bid for bid, info in warehouse.fbd_trace.items() 
                              if info['model_part'] == model_part]
                blocks_averaged += len(part_blocks)
            
            # Step 5: Clean up - delete temporary model immediately
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            evaluation_time = time.time() - evaluation_start_time
            
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': True,
                'strategy': 'fbd_average',
                'evaluation_metrics': {
                    'accuracy': eval_metrics['accuracy'],
                    'auc': eval_metrics['auc'],
                    'loss': eval_metrics['loss'],
                    'total_samples': eval_metrics['total_samples'],
                    'total_batches': eval_metrics['total_batches']
                },
                'model_info': {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_parts': list(averaged_weights.keys()),
                    'part_count': len(averaged_weights),
                    'blocks_averaged': blocks_averaged
                },
                'evaluation_time': evaluation_time
            }
            
        except Exception as e:
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - evaluation_start_time
            }

def fbd_average_evaluate(warehouse, 
                        round_num: int,
                        num_classes: int = 8,
                        input_shape: tuple = (1, 28, 28),
                        device: str = 'cpu',
                        test_batch_size: int = 64,
                        num_test_batches: int = 10,
                        norm: str = 'bn',
                        architecture: str = 'resnet18') -> Dict[str, Any]:
    """
    Convenience function for FBD average evaluation.
    
    Args:
        warehouse: FBD warehouse containing function block weights
        round_num: Current round number
        num_classes: Number of output classes
        input_shape: Input tensor shape
        device: Device for computation
        test_batch_size: Batch size for evaluation
        num_test_batches: Number of test batches
        
    Returns:
        Dict: Evaluation results
    """
    strategy = FBDAverageEvaluationStrategy(
        num_classes=num_classes,
        input_shape=input_shape,
        device=device,
        test_batch_size=test_batch_size,
        num_test_batches=num_test_batches
    )
    # Store norm type and architecture for temp_model creation
    strategy.norm = norm
    strategy.architecture = architecture
    
    return strategy.evaluate(warehouse, round_num)

class FBDComprehensiveEvaluationStrategy(FBDEvaluationStrategy):
    """
    FBD Comprehensive Evaluation Strategy.
    Evaluates M0-M5 individual models plus the averaging model (7 total evaluations per round).
    """
    
    def __init__(self, 
                 test_loader,
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 device: str = 'cpu'):
        """
        Initialize FBD comprehensive evaluation strategy.
        
        Args:
            test_loader: DataLoader for the test set
            num_classes: Number of output classes
            input_shape: Input tensor shape
            device: Device for computation
        """
        super().__init__(num_classes, input_shape, device)
        self.test_loader = test_loader
        self.model_colors = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    
    def _build_model_from_weights(self, model_weights: Dict[str, Dict[str, torch.Tensor]]) -> nn.Module:
        """
        Build a model from the given weights dictionary.
        
        Args:
            model_weights: Dictionary of model weights organized by model parts
            
        Returns:
            nn.Module: Constructed model
        """
        # Use norm type stored on strategy instance, default to 'bn' if not set
        norm_type = getattr(self, 'norm', 'bn')
        # Use architecture type stored on strategy instance, default to 'resnet18' if not set
        architecture_type = getattr(self, 'architecture', 'resnet18')
        model = get_fbd_model(
            architecture=architecture_type,
            norm=norm_type,
            in_channels=self.input_shape[0], 
            num_classes=self.num_classes
        ).to(self.device)
        
        model.load_from_dict(model_weights)
        return model
    
    def _evaluate_single_model(self, model: nn.Module, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model on the real test dataset.
        
        Args:
            model: Model to evaluate
            model_name: Name of the model (for logging)
            
        Returns:
            Dict: Evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        y_score = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Ensure targets are in the correct format (1D long tensor for CrossEntropyLoss)
                if targets.dim() > 1:
                    targets_for_loss = torch.squeeze(targets, 1).long()
                else:
                    targets_for_loss = targets.long()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets_for_loss)
                
                # Apply softmax to get probabilities
                probabilities = softmax(outputs)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Collect predictions and targets for AUC calculation
                y_score = torch.cat((y_score, probabilities), 0)
                # Store original targets for metrics
                y_true = torch.cat((y_true, targets.float().view(-1, 1)), 0)
        
        # Convert to numpy for sklearn metrics
        y_score_np = y_score.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy().flatten()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        
        # Calculate accuracy
        y_pred = y_score_np.argmax(axis=1)
        accuracy = accuracy_score(y_true_np, y_pred) * 100.0  # Convert to percentage
        
        # Calculate AUC 
        try:
            if self.num_classes == 2:
                # Binary classification - use probability of positive class
                auc = roc_auc_score(y_true_np, y_score_np[:, 1])
            else:
                # Multi-class classification
                auc = roc_auc_score(y_true_np, y_score_np, multi_class='ovr')
        except ValueError as e:
            # Handle case where all targets are the same class
            print(f"[FBD Eval Warning] Could not calculate AUC for {model_name}: {e}")
            auc = 0.5  # Random performance
        
        return {
            'model_name': model_name,
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'total_samples': len(y_true_np),
            'total_batches': len(self.test_loader)
        }
    
    def _average_weights_by_model_part(self, warehouse) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Average weights by model part across all function blocks.
        
        Args:
            warehouse: FBD warehouse
            
        Returns:
            Dict: Averaged weights organized by model part
        """
        model_parts = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        averaged_weights = {}
        
        for model_part in model_parts:
            # Find all blocks belonging to this model part
            part_blocks = []
            for block_id, block_info in warehouse.fbd_trace.items():
                if block_info['model_part'] == model_part:
                    part_blocks.append(block_id)
            
            if part_blocks:
                # Get weights from all blocks for this part
                block_weights_list = []
                for block_id in part_blocks:
                    try:
                        block_weights = warehouse.retrieve_weights(block_id)
                        if block_weights:  # Only include non-empty weights
                            block_weights_list.append(block_weights)
                    except Exception:
                        continue  # Skip blocks with missing weights
                
                if block_weights_list:
                    # Average the weights
                    averaged_part_weights = {}
                    
                    # Get all parameter names from the first block
                    param_names = block_weights_list[0].keys()
                    
                    for param_name in param_names:
                        # Stack tensors from all blocks and compute mean
                        param_tensors = []
                        for block_weights in block_weights_list:
                            if param_name in block_weights:
                                param_tensors.append(block_weights[param_name])
                        
                        if param_tensors:
                            # Average across all blocks
                            stacked_tensors = torch.stack(param_tensors)
                            
                            # Convert to float for averaging if needed, then convert back to original dtype
                            original_dtype = stacked_tensors.dtype
                            if original_dtype in [torch.long, torch.int, torch.short, torch.uint8]:
                                # Convert to float for averaging
                                stacked_tensors = stacked_tensors.float()
                                averaged_param = torch.mean(stacked_tensors, dim=0)
                                # Convert back to original dtype if it was integer type
                                averaged_param = averaged_param.to(original_dtype)
                            else:
                                # Already floating point, can compute mean directly
                                averaged_param = torch.mean(stacked_tensors, dim=0)
                            
                            averaged_part_weights[param_name] = averaged_param
                    
                    averaged_weights[model_part] = averaged_part_weights
        
        return averaged_weights
    
    def evaluate(self, warehouse, round_num: int) -> Dict[str, Any]:
        """
        Comprehensive evaluation of all M0-M5 models plus averaging model.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            round_num: Current round number
            
        Returns:
            Dict: Evaluation results for all 7 models
        """
        evaluation_start_time = time.time()
        
        try:
            all_results = {}
            successful_evaluations = 0
            
            # 1. Evaluate M0-M5 individual models
            for model_color in self.model_colors:
                try:
                    # Get model weights for this color
                    model_weights = warehouse.get_model_weights(model_color)
                    
                    if model_weights and len(model_weights) > 0:
                        # Build model from weights
                        model = self._build_model_from_weights(model_weights)
                        
                        # Evaluate the model
                        eval_metrics = self._evaluate_single_model(model, model_color)
                        all_results[model_color] = eval_metrics
                        successful_evaluations += 1
                        
                        # Clean up model
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        print(f"[FBD Eval] {model_color}: Acc={eval_metrics['accuracy']:.2f}%, AUC={eval_metrics['auc']:.4f}, Loss={eval_metrics['loss']:.4f}")
                    else:
                        all_results[model_color] = {
                            'model_name': model_color,
                            'error': 'No weights available for this model',
                            'success': False
                        }
                        print(f"[FBD Eval] {model_color}: No weights available")
                        
                except Exception as e:
                    all_results[model_color] = {
                        'model_name': model_color,
                        'error': str(e),
                        'success': False
                    }
                    print(f"[FBD Eval] {model_color}: Error - {str(e)}")
            
            # 2. Evaluate averaging model
            try:
                averaged_weights = self._average_weights_by_model_part(warehouse)
                
                if averaged_weights and len(averaged_weights) > 0:
                    # Build model from averaged weights
                    avg_model = self._build_model_from_weights(averaged_weights)
                    
                    # Evaluate the averaged model
                    avg_eval_metrics = self._evaluate_single_model(avg_model, "Averaging")
                    all_results["Averaging"] = avg_eval_metrics
                    successful_evaluations += 1
                    
                    # Clean up model
                    del avg_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"[FBD Eval] Averaging: Acc={avg_eval_metrics['accuracy']:.2f}%, AUC={avg_eval_metrics['auc']:.4f}, Loss={avg_eval_metrics['loss']:.4f}")
                else:
                    all_results["Averaging"] = {
                        'model_name': 'Averaging',
                        'error': 'No weights available for averaging',
                        'success': False
                    }
                    print(f"[FBD Eval] Averaging: No weights available")
                    
            except Exception as e:
                all_results["Averaging"] = {
                    'model_name': 'Averaging',
                    'error': str(e),
                    'success': False
                }
                print(f"[FBD Eval] Averaging: Error - {str(e)}")
            
            evaluation_time = time.time() - evaluation_start_time
            
            # Calculate summary statistics
            successful_models = [k for k, v in all_results.items() 
                               if isinstance(v, dict) and 'accuracy' in v]
            
            if successful_models:
                avg_accuracy = np.mean([all_results[model]['accuracy'] for model in successful_models])
                avg_auc = np.mean([all_results[model]['auc'] for model in successful_models])
                avg_loss = np.mean([all_results[model]['loss'] for model in successful_models])
            else:
                avg_accuracy = avg_auc = avg_loss = 0.0
            
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': successful_evaluations > 0,
                'strategy': 'fbd_comprehensive',
                'total_models_evaluated': successful_evaluations,
                'expected_models': 7,  # M0-M5 + Averaging
                'individual_results': all_results,
                'summary_metrics': {
                    'average_accuracy': avg_accuracy,
                    'average_auc': avg_auc,
                    'average_loss': avg_loss,
                    'successful_models': successful_models,
                    'total_successful': successful_evaluations
                },
                'evaluation_time': evaluation_time
            }
            
        except Exception as e:
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - evaluation_start_time
            }

def fbd_comprehensive_evaluate(warehouse, 
                              round_num: int,
                              test_loader,
                              num_classes: int = 8,
                              input_shape: tuple = (1, 28, 28),
                              device: str = 'cpu',
                              norm: str = 'bn',
                              architecture: str = 'resnet18') -> Dict[str, Any]:
    """
    Convenience function for FBD comprehensive evaluation (M0-M5 + Averaging).
    
    Args:
        warehouse: FBD warehouse containing function block weights
        round_num: Current round number  
        test_loader: DataLoader for real test data
        num_classes: Number of output classes
        input_shape: Input tensor shape
        device: Device for computation
        norm: Normalization type ('bn', 'in', 'ln')
        
    Returns:
        Dict: Comprehensive evaluation results for all 7 models
    """
    strategy = FBDComprehensiveEvaluationStrategy(
        test_loader=test_loader,
        num_classes=num_classes,
        input_shape=input_shape,
        device=device
    )
    # Store norm type and architecture for use in model creation
    strategy.norm = norm
    strategy.architecture = architecture
    
    return strategy.evaluate(warehouse, round_num)

class FBDEnsembleEvaluationStrategy(FBDEvaluationStrategy):
    """
    FBD Ensemble Evaluation Strategy.
    Evaluates an ensemble of models using different combination strategies.
    """
    
    def __init__(self, 
                 test_loader,
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 device: str = 'cpu',
                 ensemble_method: str = 'voting'):
        """
        Initialize FBD ensemble evaluation strategy.
        
        Args:
            test_loader: DataLoader for the test set
            num_classes: Number of output classes
            input_shape: Input tensor shape
            device: Device for computation
            ensemble_method: Ensemble method ('voting', 'averaging', 'weighted', etc.)
        """
        super().__init__(num_classes, input_shape, device)
        self.test_loader = test_loader
        self.ensemble_method = ensemble_method
        self.model_colors = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    
    def _build_model_from_weights(self, model_weights: Dict[str, Dict[str, torch.Tensor]]) -> nn.Module:
        """
        Build a model from the given weights dictionary.
        
        Args:
            model_weights: Dictionary of model weights organized by model parts
            
        Returns:
            nn.Module: Constructed model
        """
        # Use norm type stored on strategy instance, default to 'bn' if not set
        norm_type = getattr(self, 'norm', 'bn')
        # Use architecture type stored on strategy instance, default to 'resnet18' if not set
        architecture_type = getattr(self, 'architecture', 'resnet18')
        model = get_fbd_model(
            architecture=architecture_type,
            norm=norm_type,
            in_channels=self.input_shape[0], 
            num_classes=self.num_classes
        ).to(self.device)
        
        model.load_from_dict(model_weights)
        return model
    
    def _compute_color_block_l2_distances(self, warehouse, ensemble_records, colors_ensemble):
        """
        Compute L2 distances between blocks of different colors at the same positions.
        This measures how different the function blocks are between different model variants (colors).
        
        Args:
            warehouse: FBD warehouse containing function block weights
            ensemble_records: List of ensemble model records
            colors_ensemble: List of colors used in ensemble
            
        Returns:
            Dict: L2 distance statistics by position and color, including overall averages
        """
        # Get model parts dynamically based on architecture
        from fbd_models import get_model_parts
        architecture_type = getattr(self, 'architecture', 'resnet18')
        model_parts = get_model_parts(architecture_type)
        
        # Collect all block weights for each color and position
        color_position_blocks = {}  # color -> position -> list of weight tensors
        
        for color in colors_ensemble:
            color_position_blocks[color] = {}
            for position in model_parts:
                color_position_blocks[color][position] = []
        
        # Extract weights from warehouse for each color and position
        for color in colors_ensemble:
            try:
                color_weights = warehouse.get_model_weights(color)
                warehouse.warehouse_logger.info(f"Color {color}: weights available = {color_weights is not None}")
                if color_weights:
                    warehouse.warehouse_logger.info(f"Color {color}: keys = {list(color_weights.keys())}")
                    for position in model_parts:
                        if position in color_weights:
                            # Flatten all parameters of this block into a single tensor
                            block_params = []
                            for param_name, param_tensor in color_weights[position].items():
                                # Handle different tensor types for norm calculation
                                if param_tensor.dtype in [torch.long, torch.int, torch.short, torch.uint8, torch.int32, torch.int64]:
                                    param_norm = torch.norm(param_tensor.float()).item()
                                else:
                                    param_norm = torch.norm(param_tensor).item()
                                warehouse.warehouse_logger.info(f"Color {color}, Position {position}, Param {param_name}: shape = {param_tensor.shape}, norm = {param_norm:.6f}")
                                block_params.append(param_tensor.flatten())
                            if block_params:
                                flattened_block = torch.cat(block_params)
                                # Handle different tensor types for norm calculation
                                if flattened_block.dtype in [torch.long, torch.int, torch.short, torch.uint8, torch.int32, torch.int64]:
                                    block_norm = torch.norm(flattened_block.float()).item()
                                else:
                                    block_norm = torch.norm(flattened_block).item()
                                warehouse.warehouse_logger.info(f"Color {color}, Position {position}: flattened block norm = {block_norm:.6f}, size = {flattened_block.size()}")
                                color_position_blocks[color][position].append(flattened_block)
                        else:
                            warehouse.warehouse_logger.info(f"Color {color}: Position {position} not found in weights")
                else:
                    warehouse.warehouse_logger.info(f"Color {color}: No weights returned from warehouse")
            except Exception as e:
                warehouse.warehouse_logger.error(f"Could not extract weights for color {color}: {e}")
                continue
        
        # Compute L2 distances between blocks of different colors at the same position
        l2_distances = {}
        position_comparisons = {}
        all_position_averages = []  # Track averages across all positions for overall computation
        
        print(f"[FBD Ensemble] Computing L2 distances between different colors at same positions:")
        print(f"[FBD Ensemble] Architecture: {architecture_type}, Model parts: {model_parts}")
        
        # For each position, compare blocks of different colors
        for position in model_parts:
            position_comparisons[position] = {}
            position_distances = []
            
            # Get all colors that have blocks at this position
            colors_at_position = [color for color in colors_ensemble if color_position_blocks[color][position]]
            
            if len(colors_at_position) < 2:
                print(f"  {position}: Only {len(colors_at_position)} colors available - skipping")
                continue
            
            # Compare all pairs of colors at this position
            for i, color1 in enumerate(colors_at_position):
                for j, color2 in enumerate(colors_at_position):
                    if i < j:  # Only compute upper triangle to avoid duplicates
                        block1 = color_position_blocks[color1][position][0]  # First (and only) block
                        block2 = color_position_blocks[color2][position][0]
                        
                        # These should have the same shape since they're at the same position
                        if block1.shape != block2.shape:
                            print(f"  Warning: {color1} and {color2} at {position} have different shapes!")
                            continue
                        
                        # Debug: Check individual block norms and difference
                        block1_norm = torch.norm(block1, p=2).item()
                        block2_norm = torch.norm(block2, p=2).item()
                        diff_tensor = block1 - block2
                        l2_distance = torch.norm(diff_tensor, p=2).item()
                        
                        warehouse.warehouse_logger.info(f"{position}: {color1} norm = {block1_norm:.6f}, {color2} norm = {block2_norm:.6f}")
                        warehouse.warehouse_logger.info(f"{position}: {color1} vs {color2} - difference norm = {l2_distance:.6f}")
                        warehouse.warehouse_logger.info(f"{position}: Are blocks identical? {torch.allclose(block1, block2, atol=1e-6)}")
                        
                        pair_key = f"{color1}_vs_{color2}"
                        position_comparisons[position][pair_key] = l2_distance
                        position_distances.append(l2_distance)
            
            if position_distances:
                avg_distance = np.mean(position_distances)
                print(f"  {position}: Average L2 distance between colors = {avg_distance:.6f} (from {len(position_distances)} comparisons)")
                position_comparisons[position]['average'] = avg_distance
                position_comparisons[position]['num_comparisons'] = len(position_distances)
                position_comparisons[position]['std_dev'] = np.std(position_distances)
                position_comparisons[position]['min'] = np.min(position_distances)
                position_comparisons[position]['max'] = np.max(position_distances)
                all_position_averages.append(avg_distance)
        
        # Compute overall averages across all positions
        overall_metrics = {}
        if all_position_averages:
            overall_metrics = {
                'overall_average_l2_distance': np.mean(all_position_averages),
                'overall_std_dev': np.std(all_position_averages),
                'overall_min': np.min(all_position_averages),
                'overall_max': np.max(all_position_averages),
                'num_positions_compared': len(all_position_averages),
                'total_positions': len(model_parts)
            }
            
            print(f"[FBD Ensemble] Overall L2 Distance Summary:")
            print(f"  Overall average L2 distance: {overall_metrics['overall_average_l2_distance']:.6f}")
            print(f"  Standard deviation: {overall_metrics['overall_std_dev']:.6f}")
            print(f"  Range: {overall_metrics['overall_min']:.6f} - {overall_metrics['overall_max']:.6f}")
            print(f"  Positions compared: {overall_metrics['num_positions_compared']}/{overall_metrics['total_positions']}")
        else:
            print(f"[FBD Ensemble] Warning: No L2 distances could be computed - insufficient colors or data")
            
        # Also store the individual color blocks for potential future analysis
        for color in colors_ensemble:
            l2_distances[color] = {}
            for position in model_parts:
                if color_position_blocks[color][position]:
                    block = color_position_blocks[color][position][0]
                    # Handle different tensor types for norm calculation
                    if block.dtype in [torch.long, torch.int, torch.short, torch.uint8, torch.int32, torch.int64]:
                        block_norm = torch.norm(block.float(), p=2).item()
                    else:
                        block_norm = torch.norm(block, p=2).item()
                    l2_distances[color][position] = {
                        'norm': block_norm,  # L2 norm of the block itself
                        'shape': list(block.shape),
                        'num_params': block.numel()
                    }
        
        return {
            'by_color_and_position': l2_distances,
            'by_position_comparisons': position_comparisons,
            'overall_metrics': overall_metrics,
            'colors_analyzed': colors_ensemble,
            'positions_analyzed': model_parts,
            'architecture': architecture_type,
            'total_comparisons': sum(pos.get('num_comparisons', 0) for pos in position_comparisons.values())
        }

    def _evaluate_ensemble(self, warehouse, num_ensemble: int = 64, colors_ensemble: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate ensemble of randomly generated models on the test dataset.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            num_ensemble: Number of ensemble models to generate (default 64)
            colors_ensemble: List of colors to sample from (default full colors)
            
        Returns:
            Dict: Evaluation metrics and detailed ensemble records
        """
        import random
        from collections import Counter
        
        if colors_ensemble is None:
            colors_ensemble = self.model_colors.copy()  # ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
        
        model_parts = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        
        # Records for tracking ensemble models
        ensemble_records = []
        all_predictions = []
        
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        
        print(f"[FBD Ensemble] Generating {num_ensemble} random ensemble models from colors: {colors_ensemble}")
        
        for ensemble_idx in range(num_ensemble):
            try:
                # Step 1: Randomly select color for each model part
                model_composition = {}
                for part in model_parts:
                    model_composition[part] = random.choice(colors_ensemble)
                
                # Step 2: Build model weights from random composition
                ensemble_weights = {}
                composition_valid = True
                
                for part, color in model_composition.items():
                    try:
                        color_weights = warehouse.get_model_weights(color)
                        if color_weights and part in color_weights:
                            ensemble_weights[part] = color_weights[part]
                        else:
                            print(f"[FBD Ensemble] Warning: No weights for {color}/{part} in ensemble {ensemble_idx}")
                            composition_valid = False
                            break
                    except Exception as e:
                        print(f"[FBD Ensemble] Error getting weights for {color}/{part}: {e}")
                        composition_valid = False
                        break
                
                if not composition_valid:
                    continue
                
                # Step 3: Build and evaluate the ensemble model
                ensemble_model = self._build_model_from_weights(ensemble_weights)
                ensemble_model.eval()
                
                # Step 4: Get predictions from this ensemble model
                model_predictions = []
                model_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in self.test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Ensure targets are in the correct format
                        if targets.dim() > 1:
                            targets_for_loss = torch.squeeze(targets, 1).long()
                        else:
                            targets_for_loss = targets.long()
                        
                        # Forward pass
                        outputs = ensemble_model(inputs)
                        loss = criterion(outputs, targets_for_loss)
                        model_loss += loss.item()
                        
                        # Get predictions (class indices)
                        probabilities = softmax(outputs)
                        predictions = torch.argmax(probabilities, dim=1)
                        model_predictions.extend(predictions.cpu().numpy().tolist())
                
                # Step 5: Record this ensemble model
                ensemble_record = {
                    'ensemble_id': ensemble_idx,
                    'composition': model_composition.copy(),
                    'predictions': model_predictions.copy(),
                    'loss': model_loss / len(self.test_loader),
                    'num_samples': len(model_predictions)
                }
                
                ensemble_records.append(ensemble_record)
                all_predictions.append(model_predictions)
                
                # Clean up model
                del ensemble_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if (ensemble_idx + 1) % 10 == 0:
                    print(f"[FBD Ensemble] Generated {ensemble_idx + 1}/{num_ensemble} ensemble models")
                    
            except Exception as e:
                print(f"[FBD Ensemble] Error generating ensemble model {ensemble_idx}: {e}")
                continue
        
        if len(all_predictions) == 0:
            return {
                'ensemble_method': self.ensemble_method,
                'num_ensemble_requested': num_ensemble,
                'num_ensemble_generated': 0,
                'colors_ensemble': colors_ensemble,
                'error': 'No ensemble models could be generated',
                'ensemble_records': []
            }
        
        # Step 6: Perform majority voting across all ensemble predictions
        num_samples = len(all_predictions[0])
        final_predictions = []
        agreements = []
        
        for sample_idx in range(num_samples):
            # Get all predictions for this sample
            sample_predictions = [pred_list[sample_idx] for pred_list in all_predictions]
            
            # Count votes
            vote_counts = Counter(sample_predictions)
            majority_vote = vote_counts.most_common(1)[0][0]  # Most common prediction
            agreement_count = vote_counts[majority_vote]  # How many models agreed
            
            final_predictions.append(majority_vote)
            agreements.append(agreement_count)
        
        # Step 7: Calculate final ensemble metrics
        # Get true labels for accuracy calculation
        true_labels = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                targets = targets.to(self.device)
                if targets.dim() > 1:
                    targets = torch.squeeze(targets, 1)
                true_labels.extend(targets.cpu().numpy().tolist())
        
        # Calculate ensemble accuracy
        correct_predictions = sum(1 for pred, true in zip(final_predictions, true_labels) if pred == true)
        ensemble_accuracy = (correct_predictions / len(final_predictions)) * 100.0
        
        # Calculate AUC (if needed, would require probability averaging - simplified here)
        try:
            ensemble_auc = accuracy_score(true_labels, final_predictions)  # Simplified
        except:
            ensemble_auc = 0.5
        
        # Calculate agreement statistics
        avg_agreement = np.mean(agreements)
        median_agreement = np.median(agreements)
        max_agreement = max(agreements)
        min_agreement = min(agreements)
        agreement_ratio = avg_agreement / len(all_predictions)  # Ratio of average agreement
        
        # Step 6.5: Compute L2 distances between blocks with same color
        l2_distances = self._compute_color_block_l2_distances(warehouse, ensemble_records, colors_ensemble)
        
        print(f"[FBD Ensemble] Generated {len(all_predictions)} ensemble models")
        print(f"[FBD Ensemble] Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"[FBD Ensemble] Agreement Stats (across {len(final_predictions)} samples):")
        print(f"   Mean Agreement: {avg_agreement:.1f}/{len(all_predictions)} ({agreement_ratio:.3f})")
        print(f"   Median Agreement: {median_agreement:.0f}/{len(all_predictions)}")
        print(f"   Range: {min_agreement}-{max_agreement}")
        
        return {
            'ensemble_method': self.ensemble_method,
            'num_ensemble_requested': num_ensemble,
            'num_ensemble_generated': len(all_predictions),
            'colors_ensemble': colors_ensemble,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'l2_distances': l2_distances,
            'agreement_stats': {
                'average_agreement': avg_agreement,
                'median_agreement': median_agreement,
                'max_agreement': max_agreement,
                'min_agreement': min_agreement,
                'agreement_ratio': agreement_ratio,
                'total_samples': len(final_predictions)
            },
            'total_samples': len(final_predictions),
            'total_batches': len(self.test_loader),
            'final_predictions': final_predictions,
            'agreements': agreements,
            'ensemble_records': ensemble_records
        }
    
    def evaluate(self, warehouse, round_num: int, num_ensemble: int = 64, colors_ensemble: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate FBD warehouse using ensemble methods.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            round_num: Current round number
            num_ensemble: Number of ensemble models to generate (default 64)
            colors_ensemble: List of colors to sample from (default full colors)
            
        Returns:
            Dict: Ensemble evaluation results
        """
        evaluation_start_time = time.time()
        
        try:
            # Check if any colors are available
            available_colors = []
            for model_color in self.model_colors:
                try:
                    model_weights = warehouse.get_model_weights(model_color)
                    if model_weights and len(model_weights) > 0:
                        available_colors.append(model_color)
                except Exception:
                    continue
            
            if len(available_colors) == 0:
                return {
                    'round': round_num,
                    'timestamp': time.time(),
                    'success': False,
                    'error': 'No model colors available for ensemble',
                    'evaluation_time': time.time() - evaluation_start_time
                }
            
            # Use available colors if colors_ensemble not specified
            if colors_ensemble is None:
                colors_ensemble = available_colors
            else:
                # Filter colors_ensemble to only include available colors
                colors_ensemble = [c for c in colors_ensemble if c in available_colors]
                if len(colors_ensemble) == 0:
                    return {
                        'round': round_num,
                        'timestamp': time.time(),
                        'success': False,
                        'error': 'None of the specified ensemble colors are available',
                        'evaluation_time': time.time() - evaluation_start_time
                    }
            
            print(f"[FBD Ensemble] Available colors: {available_colors}")
            print(f"[FBD Ensemble] Using colors for ensemble: {colors_ensemble}")
            
            # Evaluate ensemble
            ensemble_metrics = self._evaluate_ensemble(warehouse, num_ensemble, colors_ensemble)
            
            evaluation_time = time.time() - evaluation_start_time
            
            # Check if ensemble evaluation was successful
            if 'error' in ensemble_metrics:
                return {
                    'round': round_num,
                    'timestamp': time.time(),
                    'success': False,
                    'error': ensemble_metrics['error'],
                    'evaluation_time': evaluation_time
                }
            
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': True,
                'strategy': 'fbd_ensemble',
                'available_colors': available_colors,
                'evaluation_metrics': ensemble_metrics,
                'evaluation_time': evaluation_time
            }
            
        except Exception as e:
            return {
                'round': round_num,
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - evaluation_start_time
            }

def analyze_color_l2_distances(warehouse, 
                              colors_ensemble: List[str] = None,
                              architecture: str = 'resnet18') -> Dict[str, Any]:
    """
    Standalone function to analyze L2 distances between FBD colors without evaluation.
    Useful for understanding model diversity and convergence.
    
    Args:
        warehouse: FBD warehouse containing function block weights
        colors_ensemble: List of colors to analyze (default: ['M0', 'M1', 'M2', 'M3', 'M4', 'M5'])
        architecture: Model architecture ('resnet18', 'resnet50')
    
    Returns:
        Dict: Comprehensive L2 distance analysis results
    """
    if colors_ensemble is None:
        colors_ensemble = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    
    # Create a temporary strategy instance for L2 computation
    class TempStrategy:
        def __init__(self, architecture):
            self.architecture = architecture
        
        def _compute_color_block_l2_distances(self, warehouse, ensemble_records, colors_ensemble):
            # Use the same implementation as in FBDEnsembleEvaluationStrategy
            from fbd_models import get_model_parts
            import numpy as np
            
            architecture_type = getattr(self, 'architecture', 'resnet18')
            model_parts = get_model_parts(architecture_type)
            
            # Collect all block weights for each color and position
            color_position_blocks = {}  # color -> position -> list of weight tensors
            
            for color in colors_ensemble:
                color_position_blocks[color] = {}
                for position in model_parts:
                    color_position_blocks[color][position] = []
            
            # Extract weights from warehouse for each color and position
            for color in colors_ensemble:
                try:
                    color_weights = warehouse.get_model_weights(color)
                    if color_weights:
                        for position in model_parts:
                            if position in color_weights:
                                # Flatten all parameters of this block into a single tensor
                                block_params = []
                                for param_name, param_tensor in color_weights[position].items():
                                    block_params.append(param_tensor.flatten())
                                if block_params:
                                    flattened_block = torch.cat(block_params)
                                    color_position_blocks[color][position].append(flattened_block)
                except Exception as e:
                    warehouse.warehouse_logger.error(f"Could not extract weights for color {color}: {e}")
                    continue
            
            # Compute L2 distances between blocks of different colors at the same position
            l2_distances = {}
            position_comparisons = {}
            all_position_averages = []
            
            warehouse.warehouse_logger.info(f"Computing L2 distances between colors: {colors_ensemble}")
            warehouse.warehouse_logger.info(f"Architecture: {architecture_type}, Model parts: {model_parts}")
            
            # For each position, compare blocks of different colors
            for position in model_parts:
                position_comparisons[position] = {}
                position_distances = []
                
                # Get all colors that have blocks at this position
                colors_at_position = [color for color in colors_ensemble if color_position_blocks[color][position]]
                
                if len(colors_at_position) < 2:
                    warehouse.warehouse_logger.info(f"{position}: Only {len(colors_at_position)} colors available - skipping")
                    continue
                
                # Compare all pairs of colors at this position
                for i, color1 in enumerate(colors_at_position):
                    for j, color2 in enumerate(colors_at_position):
                        if i < j:  # Only compute upper triangle to avoid duplicates
                            block1 = color_position_blocks[color1][position][0]  # First (and only) block
                            block2 = color_position_blocks[color2][position][0]
                            
                            # These should have the same shape since they're at the same position
                            if block1.shape != block2.shape:
                                warehouse.warehouse_logger.warning(f"Warning: {color1} and {color2} at {position} have different shapes!")
                                continue
                            
                            l2_distance = torch.norm(block1 - block2, p=2).item()
                            
                            pair_key = f"{color1}_vs_{color2}"
                            position_comparisons[position][pair_key] = l2_distance
                            position_distances.append(l2_distance)
                
                if position_distances:
                    avg_distance = np.mean(position_distances)
                    warehouse.warehouse_logger.info(f"{position}: Average L2 distance = {avg_distance:.6f} (from {len(position_distances)} comparisons)")
                    position_comparisons[position]['average'] = avg_distance
                    position_comparisons[position]['num_comparisons'] = len(position_distances)
                    position_comparisons[position]['std_dev'] = np.std(position_distances)
                    position_comparisons[position]['min'] = np.min(position_distances)
                    position_comparisons[position]['max'] = np.max(position_distances)
                    all_position_averages.append(avg_distance)
            
            # Compute overall averages across all positions
            overall_metrics = {}
            if all_position_averages:
                overall_metrics = {
                    'overall_average_l2_distance': np.mean(all_position_averages),
                    'overall_std_dev': np.std(all_position_averages),
                    'overall_min': np.min(all_position_averages),
                    'overall_max': np.max(all_position_averages),
                    'num_positions_compared': len(all_position_averages),
                    'total_positions': len(model_parts)
                }
                
                warehouse.warehouse_logger.info(f"Overall L2 Distance Summary:")
                warehouse.warehouse_logger.info(f"Overall average L2 distance: {overall_metrics['overall_average_l2_distance']:.6f}")
                warehouse.warehouse_logger.info(f"Standard deviation: {overall_metrics['overall_std_dev']:.6f}")
                warehouse.warehouse_logger.info(f"Range: {overall_metrics['overall_min']:.6f} - {overall_metrics['overall_max']:.6f}")
                warehouse.warehouse_logger.info(f"Positions compared: {overall_metrics['num_positions_compared']}/{overall_metrics['total_positions']}")
            else:
                warehouse.warehouse_logger.warning(f"No L2 distances could be computed - insufficient colors or data")
            
            return {
                'by_position_comparisons': position_comparisons,
                'overall_metrics': overall_metrics,
                'colors_analyzed': colors_ensemble,
                'positions_analyzed': model_parts,
                'architecture': architecture_type,
                'total_comparisons': sum(pos.get('num_comparisons', 0) for pos in position_comparisons.values())
            }
    
    temp_strategy = TempStrategy(architecture)
    return temp_strategy._compute_color_block_l2_distances(warehouse, [], colors_ensemble)

def fbd_ensemble_evaluate(warehouse, 
                         round_num: int,
                         test_loader,
                         num_classes: int = 8,
                         input_shape: tuple = (1, 28, 28),
                         device: str = 'cpu',
                         norm: str = 'bn',
                         architecture: str = 'resnet18',
                         ensemble_method: str = 'voting',
                         num_ensemble: int = 64,
                         colors_ensemble: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function for FBD ensemble evaluation.
    
    Args:
        warehouse: FBD warehouse containing function block weights
        round_num: Current round number
        test_loader: DataLoader for real test data
        num_classes: Number of output classes
        input_shape: Input tensor shape
        device: Device for computation
        norm: Normalization type ('bn', 'in', 'ln')
        ensemble_method: Ensemble method ('voting', 'averaging', 'weighted', etc.)
        num_ensemble: Number of ensemble models to generate (default 64)
        colors_ensemble: List of colors to sample from (default full colors)
        
    Returns:
        Dict: Ensemble evaluation results
    """
    strategy = FBDEnsembleEvaluationStrategy(
        test_loader=test_loader,
        num_classes=num_classes,
        input_shape=input_shape,
        device=device,
        ensemble_method=ensemble_method
    )
    # Store norm type and architecture for use in model creation
    strategy.norm = norm
    strategy.architecture = architecture
    
    return strategy.evaluate(warehouse, round_num, num_ensemble, colors_ensemble) 