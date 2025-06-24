"""
FBD Federated Learning Main - Flower Integration
Integrates FBD (Function Block Diversification) with Flower federated learning framework
"""

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import argparse
import flwr as fl
import torch
import os
import json
import time
import random
import numpy as np
import logging
from flwr.client import Client
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from collections import OrderedDict
import pickle
from typing import Dict, List, Tuple, Optional
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar

# Import existing components
from dataset import load_data, partition_data, get_data_loader
from models import ResNet18_FBD_BN, ResNet18_FBD_IN, ResNet18_FBD_LN
from medmnist import INFO
from config_loader import load_config

# Import FBD components
from fbd_logic import load_fbd_settings, load_shipping_plan, load_request_plan, FBDWarehouse, generate_client_model_palettes
from fbd_eval_strategy import fbd_average_evaluate, fbd_comprehensive_evaluate, fbd_ensemble_evaluate
from fbd_communication import WeightTransfer

# Import pretrained weight loader
from fbd_root_ckpt import get_pretrained_fbd_model


def get_resnet18_fbd_model(norm: str, in_channels: int, num_classes: int, use_imagenet: bool = False, device: str = 'cpu'):
    """Get the appropriate ResNet18 FBD model based on normalization type."""
    if use_imagenet:
        # Use ImageNet pretrained weights
        logging.info(f"🔄 Loading ResNet18 FBD with ImageNet pretrained weights ({norm.upper()} normalization)")
        return get_pretrained_fbd_model(
            architecture='resnet18',
            norm=norm,
            in_channels=in_channels,
            num_classes=num_classes,
            device=device,
            use_pretrained=True
        )
    else:
        # Use random initialization
        if norm == 'bn':
            return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)
        elif norm == 'in':
            return ResNet18_FBD_IN(in_channels=in_channels, num_classes=num_classes)
        elif norm == 'ln':
            return ResNet18_FBD_LN(in_channels=in_channels, num_classes=num_classes)
        else:
            # Default to batch normalization if norm type is not specified or unknown
            return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)


class FBDFlowerClient(fl.client.Client):
    """FBD-enabled Flower client that integrates with FBD warehouse system."""
    
    def __init__(self, cid, model, train_loader, val_loader, test_loader, data_flag, device, 
                 fbd_config_path, communication_dir, client_palette):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_flag = data_flag
        self.device = device
        
        # FBD specific attributes
        self.fbd_config_path = fbd_config_path
        self.communication_dir = communication_dir
        self.client_palette = client_palette
        
        # Initialize FBD communication
        self.communication = WeightTransfer(communication_dir)
        
        # Load FBD settings
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        
        logging.info(f"[FBD Client {cid}] Initialized with {len(client_palette)} FBD blocks")

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        """Extract model parameters."""
        ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        parameters = ndarrays_to_parameters(ndarrays)
        return fl.common.GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"), 
            parameters=parameters
        )

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Perform FBD federated training."""
        config = ins.config
        round_num = config.get("server_round", 1)
        local_lr = config.get("local_learning_rate", 0.001)
        
        # FBD: Receive weights from server (shipping phase)
        try:
            received_weights = self.communication.client_receive_weights(self.cid, round_num)
            if received_weights:
                logging.info(f"[FBD Client {self.cid}] Round {round_num}: Received {len(received_weights)} model parts")
                self.model.load_from_dict(received_weights)
        except (TimeoutError, FileNotFoundError) as e:
            logging.info(f"[FBD Client {self.cid}] Round {round_num}: No weights received from server - using current model")
        
        # Perform local training
        current_update_plan = config.get("current_update_plan", None)
        train_result = self._train_model(local_lr, current_update_plan=current_update_plan, round_num=round_num)
        
        # Handle different return types from train function
        if isinstance(train_result, tuple):
            train_loss, regularizer_metrics = train_result
        else:
            train_loss = train_result
            regularizer_metrics = None
        
        # Evaluate after training
        train_loss, train_auc, train_acc = self._test_model(self.train_loader)
        val_loss, val_auc, val_acc = self._test_model(self.val_loader)
        
        logging.info(f"[FBD Client {self.cid}] Round {round_num}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # FBD: Send updated weights to warehouse (always send trained weights)
        # Extract weights according to client palette for this round
        if self.client_palette:
            extracted_weights = {}
            for block_id, block_info in self.client_palette.items():
                model_part = block_info['model_part']
                # Extract weights for this model part
                part_weights = self.model.send_for_dict([model_part])
                if part_weights:
                    extracted_weights[block_id] = part_weights[model_part]
            
            # Send all client's trained weights to warehouse
            if extracted_weights:
                self.communication.client_send_weights(self.cid, round_num, extracted_weights, list(extracted_weights.keys()))
                logging.info(f"[FBD Client {self.cid}] Sent {len(extracted_weights)} trained FBD blocks to server")
        
        # Note: Collection phase will be handled by server using these weights
        
        # Return metrics (no parameters needed for FBD - using file communication)
        metrics_dict = {
            "train_loss": train_loss,
            "train_auc": train_auc, 
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc
        }
        
        # Add regularizer metrics if available
        if regularizer_metrics is not None:
            metrics_dict.update({
                "regularizer_type": regularizer_metrics.get('regularizer_type'),
                "num_regularizers": regularizer_metrics.get('num_regularizers', 0),
                "regularization_strength": regularizer_metrics.get('regularization_strength', 0.0),
                "avg_regularizer_distance": regularizer_metrics.get('avg_regularizer_distance', 0.0),
                "max_regularizer_distance": regularizer_metrics.get('max_regularizer_distance', 0.0),
                "min_regularizer_distance": regularizer_metrics.get('min_regularizer_distance', 0.0),
                "std_regularizer_distance": regularizer_metrics.get('std_regularizer_distance', 0.0),
                "regularizer_batch_details": regularizer_metrics.get('regularizer_distances', [])
            })
        
        # Return empty parameters since FBD uses file-based communication
        empty_params = ndarrays_to_parameters([np.array([0.0])])
        
        return FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=empty_params,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics_dict
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """Evaluate model on validation set."""
        loss, auc, acc = self._test_model(self.val_loader)
        
        metrics = {"loss": float(loss), "auc": float(auc), "acc": float(acc)}
        
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.val_loader.dataset),
            metrics=metrics,
        )

    def _train_model(self, lr, current_update_plan=None, round_num=None):
        """Train the model locally."""
        from client import train  # Use existing training function
        return train(self.model, self.train_loader, epochs=1, device=self.device, 
                    data_flag=self.data_flag, lr=lr, current_update_plan=current_update_plan,
                    client_id=self.cid, round_num=round_num)

    def _test_model(self, data_loader):
        """Test the model."""
        from client import test  # Use existing test function
        return test(self.model, data_loader, device=self.device, data_flag=self.data_flag)

    def _extract_fbd_weights(self, request_list):
        """Extract FBD weights according to request list and client palette."""
        extracted_weights = {}
        
        for block_id in request_list:
            if block_id in self.client_palette:
                model_part = self.client_palette[block_id]['model_part']
                # Extract weights for this model part
                part_weights = self.model.send_for_dict([model_part])
                if part_weights:
                    extracted_weights[block_id] = part_weights[model_part]
        
        return extracted_weights


class FBDStrategy(FedAvg):
    """FBD-enabled Flower strategy that manages FBD warehouse and evaluation."""
    
    def __init__(self, fbd_config_path, shipping_plan_path, request_plan_path, 
                 num_clients, communication_dir, model_template, output_dir, 
                 num_classes, input_shape, test_dataset, batch_size, norm_type='bn', 
                 num_rounds=1, num_ensemble=64, ensemble_colors=None, 
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
        import os
        self.update_plan = None
        if update_plan_path and os.path.exists(update_plan_path):
            try:
                import json
                with open(update_plan_path, 'r') as f:
                    self.update_plan = json.load(f)
                logging.info(f"[FBD Strategy] Loaded update plan from {update_plan_path}")
            except Exception as e:
                logging.warning(f"[FBD Strategy] Failed to load update plan: {e}")
        else:
            logging.info(f"[FBD Strategy] No update plan provided or file not found")
        
        # Initialize FBD warehouse
        self.warehouse = FBDWarehouse(self.fbd_trace, model_template)
        
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
        import time
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
            
            # Add client-specific update plan if available
            if (self.update_plan is not None and 
                str(server_round) in self.update_plan and 
                str(client_id) in self.update_plan[str(server_round)]):
                
                client_update_plan = self.update_plan[str(server_round)][str(client_id)]
                base_config["current_update_plan"] = client_update_plan
                
                logging.info(f"[FBD Strategy] Sending update plan to client {client_id}: "
                           f"model_to_update with {len(client_update_plan['model_as_regularizer'])} regularizers")
            
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
            norm=self.norm_type  # Use stored norm type
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
                
                # Log L2 distance summary
                position_comparisons = l2_distances.get('by_position_comparisons', {})
                if position_comparisons:
                    logging.info(f"[FBD Strategy] Round {server_round} L2 Distance Summary:")
                    for position, comparisons in position_comparisons.items():
                        avg_distance = comparisons.get('average', 0)
                        if avg_distance > 0:
                            logging.info(f"  {position}: Average L2 distance = {avg_distance:.6f}")
                    
                    # Log overall statistics
                    all_avg_distances = [comp.get('average', 0) for comp in position_comparisons.values() if comp.get('average', 0) > 0]
                    if all_avg_distances:
                        overall_avg = sum(all_avg_distances) / len(all_avg_distances)
                        logging.info(f"  Overall average L2 distance across positions: {overall_avg:.6f}")
        
        # Track round end time
        import time
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
        import time
        
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
        
        import csv
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


def get_fit_config_fn(config):
    """Return a function which returns training configurations with learning rate scheduling."""
    def fit_config(server_round: int):
        # Learning rate schedule:
        # Rounds 1-20: original learning rate
        # Rounds 21-50: learning rate / 10
        # Rounds 51+: learning rate / 20
        
        base_lr = config.local_learning_rate
        
        if server_round <= 20:
            current_lr = base_lr
            schedule_info = "base rate"
        elif server_round <= 50:
            current_lr = base_lr / 10
            schedule_info = "1/10 rate"
        else:
            current_lr = base_lr / 20
            schedule_info = "1/20 rate"
        
        return {
            "local_learning_rate": current_lr,
            "server_round": server_round,
            "lr_schedule_info": schedule_info,
            "base_learning_rate": base_lr
        }
    return fit_config


def main():
    parser = argparse.ArgumentParser(description="FBD Federated Learning with Flower")
    parser.add_argument("--dataset", type=str, default="bloodmnist", help="MedMNIST dataset name")
    parser.add_argument("--model_flag", type=str, default="resnet18_fbd", help="Model to train")
    parser.add_argument("--size", type=int, default=28, help="Image size")
    parser.add_argument("--iid", action="store_true", help="Whether to partition data in an IID manner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--cpus_per_client", type=int, default=6, help="Number of CPUs allocated per client")
    parser.add_argument("--imagenet", action="store_true", help="Use ImageNet pretrained weights for initialization")
    
    # FBD specific arguments
    parser.add_argument("--fbd_config", type=str, default="fbd_record/fbd_settings.py", 
                       help="Path to FBD configuration file")
    parser.add_argument("--shipping_plan", type=str, default="shipping_plan.json",
                       help="Path to shipping plan JSON file")
    parser.add_argument("--request_plan", type=str, default="request_plan.json", 
                       help="Path to request plan JSON file")
    parser.add_argument("--update_plan", type=str, default="fbd_record/update_plan.json",
                       help="Path to update plan JSON file")
    parser.add_argument("--communication_dir", type=str, default="fbd_comm",
                       help="Directory for FBD communication files")
    parser.add_argument("--ensemble_size", type=int, default=None,
                       help="Number of ensemble models to generate (overrides config)")
    parser.add_argument("--ensemble_colors", type=str, nargs='+', default=None,
                       help="List of model colors for ensemble (e.g., M1 M2)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.dataset, args.model_flag.replace("_fbd", ""), args.size)
    
    # Override ensemble settings if provided via command line
    if args.ensemble_size is not None:
        config.num_ensemble = args.ensemble_size
    
    # Store original config rounds before any modifications
    original_config_rounds = config.num_rounds
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir is None:
        output_dir = os.path.join("runs", "fbd", args.dataset, str(args.size), str(args.seed))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Setup Logging ---
    log_file_path = os.path.join(output_dir, "simulation.log")
    
    # Configure the root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'),
                            logging.StreamHandler()
                        ])

    # Load model and data
    info = INFO[args.dataset]
    n_channels = 3 if config.as_rgb else info['n_channels']
    n_classes = len(info['label'])
    
    # Get normalization type from config (default to 'bn' if not specified)
    norm_type = getattr(config, 'norm', 'bn')
    
    # Create model with optional ImageNet pretraining
    if args.imagenet:
        logging.info(f"🎯 ImageNet pretraining enabled for server model")
        model = get_resnet18_fbd_model(norm_type, n_channels, n_classes, use_imagenet=True, device=device)
    else:
        logging.info(f"🎯 Using random initialization for server model")
        model = get_resnet18_fbd_model(norm_type, n_channels, n_classes, use_imagenet=False, device=device)
        model = model.to(device)
    
    # Load REGULARIZER_PARAMS for configuration saving
    regularizer_params = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fbd_config", args.fbd_config)
        fbd_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fbd_config_module)
        regularizer_params = getattr(fbd_config_module, 'REGULARIZER_PARAMS', {})
    except Exception as e:
        logging.warning(f"Failed to load REGULARIZER_PARAMS for config: {e}")
    
    # Save configuration
    config_dict = {
        "dataset": args.dataset,
        "model_flag": args.model_flag,
        "normalization": norm_type,
        "actual_model_class": model.__class__.__name__,
        "imagenet_pretrained": args.imagenet,
        "initialization": "ImageNet pretrained" if args.imagenet else "Random initialization",
        "image_size": config.size,
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "num_rounds_note": "Derived from shipping plan, not config file",
        "original_config_rounds": original_config_rounds,
        "regularizer_params": regularizer_params,
        "batch_size": config.batch_size,
        "local_learning_rate": config.local_learning_rate,
        "num_ensemble": config.num_ensemble,
        "iid_partitioning": args.iid,
        "random_seed": args.seed,
        "output_directory": output_dir,
        "device": str(device),
        "fbd_config": args.fbd_config,
        "shipping_plan": args.shipping_plan,
        "request_plan": args.request_plan,
        "communication_dir": args.communication_dir
    }
    
    with open(os.path.join(output_dir, "fbd_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    
    logging.info(f"\n--- FBD Federated Learning Configuration ---")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.model_flag}")
    logging.info(f"Normalization: {norm_type}")
    logging.info(f"Actual Model Class: {model.__class__.__name__}")
    logging.info(f"Initialization: {'ImageNet pretrained' if args.imagenet else 'Random initialization'}")
    logging.info(f"Clients: {config.num_clients}")
    logging.info(f"Rounds: {config.num_rounds} (from shipping plan)")
    logging.info(f"Ensemble Models: {config.num_ensemble}")
    logging.info(f"Device: {device}")
    logging.info(f"FBD Config: {args.fbd_config}")
    logging.info(f"Output: {output_dir}")
    logging.info("--------------------------------------------\n")
    
    # Log the actual model being used
    logging.info(f"✅ Model instantiated: {model.__class__.__name__}")
    logging.info(f"   Normalization type: {norm_type}")
    logging.info(f"   Input channels: {n_channels}")
    logging.info(f"   Output classes: {n_classes}")
    logging.info(f"   Initialization: {'ImageNet pretrained' if args.imagenet else 'Random initialization'}")
    logging.info("")
    
    # Load and partition data
    train_dataset, val_dataset, test_dataset = load_data(
        args.dataset, config.resize, config.as_rgb, config.download, config.size
    )
    client_datasets = partition_data(train_dataset, config.num_clients, iid=args.iid, data_flag=args.dataset)
    
    # Load FBD settings and generate client palettes
    client_palettes = generate_client_model_palettes(config.num_clients, args.fbd_config)
    
    # Clean up communication directory
    if os.path.exists(args.communication_dir):
        import shutil
        shutil.rmtree(args.communication_dir)
    os.makedirs(args.communication_dir, exist_ok=True)
    
    # Load shipping, request, and update plans
    logging.info("Loading shipping, request, and update plans...")
    shipping_plan = load_shipping_plan(args.shipping_plan)
    request_plan = load_request_plan(args.request_plan)
    
    # Load update plan
    update_plan = None
    if args.update_plan and os.path.exists(args.update_plan):
        with open(args.update_plan, 'r') as f:
            update_plan = json.load(f)
    
    # FBD training is driven by shipping plan, not config
    max_shipping_round = max(shipping_plan.keys()) if shipping_plan else 0
    max_request_round = max(request_plan.keys()) if request_plan else 0
    fbd_total_rounds = max(max_shipping_round, max_request_round)
    
    if fbd_total_rounds <= 0:
        raise ValueError("No valid shipping/request plan found - cannot determine number of rounds")
    
    # Always use shipping plan rounds (ignore config.num_rounds completely)
    config.num_rounds = fbd_total_rounds
    
    logging.info(f"📋 FBD Training Configuration:")
    logging.info(f"   Shipping plan rounds: {len(shipping_plan)}")
    logging.info(f"   Request plan rounds: {len(request_plan)}")
    logging.info(f"   FBD training rounds: {config.num_rounds} (authoritative)")
    logging.info(f"   Config rounds (ignored): {original_config_rounds}")
    logging.info(f"   ✓ Training will stop after {config.num_rounds} rounds as defined by shipping plan")
    
    logging.info(f"Loaded shipping plan for {len(shipping_plan)} rounds")
    logging.info(f"Loaded request plan for {len(request_plan)} rounds")
    
    def client_fn(cid: str) -> Client:
        """Create an FBD Flower client."""
        client_dataset = client_datasets[int(cid)]
        train_loader = get_data_loader(client_dataset, config.batch_size)
        val_loader = get_data_loader(val_dataset, config.batch_size)
        test_loader = get_data_loader(test_dataset, config.batch_size)
        
        # Create client-specific model (clients use same initialization as server)
        if args.imagenet:
            client_model = get_resnet18_fbd_model(norm_type, n_channels, n_classes, use_imagenet=True, device=device)
        else:
            client_model = get_resnet18_fbd_model(norm_type, n_channels, n_classes, use_imagenet=False, device=device)
            client_model = client_model.to(device)
        
        # Get client palette
        client_palette = client_palettes.get(int(cid), {})
        
        return FBDFlowerClient(
            cid=int(cid),
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            data_flag=args.dataset,
            device=device,
            fbd_config_path=args.fbd_config,
            communication_dir=args.communication_dir,
            client_palette=client_palette
        ).to_client()
    
    # Initialize FBD strategy
    strategy = FBDStrategy(
        fbd_config_path=args.fbd_config,
        shipping_plan_path=args.shipping_plan,
        request_plan_path=args.request_plan,
        update_plan_path=args.update_plan,  # Pass update plan path
        num_clients=config.num_clients,
        communication_dir=args.communication_dir,
        model_template=model,
        output_dir=output_dir,
        num_classes=n_classes,
        input_shape=(n_channels, config.size, config.size),
        test_dataset=test_dataset,
        batch_size=config.batch_size,
        norm_type=norm_type,  # Pass normalization type
        num_rounds=config.num_rounds,  # Pass total number of rounds
        num_ensemble=config.num_ensemble,  # Pass number of ensemble models from config
        ensemble_colors=args.ensemble_colors,  # Pass ensemble colors from command line
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Disable client-side evaluation
        min_fit_clients=config.num_clients,
        min_evaluate_clients=0,  # Disable client-side evaluation
        min_available_clients=config.num_clients,
        on_fit_config_fn=get_fit_config_fn(config)
    )
    
    # Define client resources
    gpus_per_client = 1 / (config.num_clients + 1) if device.type == "cuda" else 0
    client_resources = {"num_cpus": args.cpus_per_client, "num_gpus": gpus_per_client}
    
    logging.info(f"🚀 Starting FBD Federated Learning Simulation")
    logging.info(f"Clients: {config.num_clients}, Rounds: {config.num_rounds}")
    logging.info(f"GPU per client: {gpus_per_client:.3f}")
    logging.info(f"📄 Training progress will be saved to JSON files instead of verbose console output")
    
    # Temporarily reduce Flower's logging verbosity during simulation
    fl_logger = logging.getLogger("flwr")
    original_level = fl_logger.level
    fl_logger.setLevel(logging.WARNING)  # Suppress Flower's verbose output
    
    try:
        # Start Flower simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )
    finally:
        # Restore original logging level
        fl_logger.setLevel(original_level)
    
    # Save final results after training completion
    strategy.save_final_results()
    
    logging.info(f"\n🎉 FBD Federated Learning Completed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"📊 Best performance summary: {output_dir}/best_model_performance.csv")
    logging.info(f"💾 Best warehouse weights: {output_dir}/best_warehouse_weights.json")


if __name__ == "__main__":
    main() 