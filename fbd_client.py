"""
FBD Client Implementation for Federated Learning

This module contains all FBD client-related functionality for production federated learning.

Classes:
    - FBDFlowerClient: Main Flower-based client for FBD federated learning

Functions:
    - train(): Model training with FBD regularization support
    - test(): Model evaluation  
    - _extract_model_parts(): Helper for FBD block extraction
    - _load_regularizer_models(): Helper for loading regularizer models
    - _compute_weight_regularizer(): Weight-based regularization computation
    - _compute_consistency_regularizer(): Output consistency regularization computation
    - _update_main_model_from_parts(): Model update helper

This module provides the core client-side functionality for FBD federated learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import time
import os
import logging
from collections import OrderedDict
import flwr as fl
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import json

from fbd_models import get_fbd_model
from tests.test_trainer import LocalTrainer
from fbd_communication import WeightTransfer
from fbd_utils import load_fbd_settings
from medmnist import INFO
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Import regularizer configuration
from fbd_record.fbd_settings import REGULARIZER_PARAMS


# ====================================================================================
# TRAINING AND TESTING FUNCTIONS (moved from client.py)
# ====================================================================================

def train(model, train_loader, epochs, device, data_flag, lr, current_update_plan=None, client_id=None, round_num=None, client_logger=None):
    """Train the model on the training set."""
    info = INFO[data_flag]
    task = info['task']
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    # Check if we have current update plan for regularized training
    use_update_plan = (current_update_plan is not None and 
                      client_id is not None and 
                      round_num is not None)
    
    # Initialize regularizer metrics tracking
    regularizer_metrics = {
        'regularizer_distances': [],
        'regularizer_type': None,
        'num_regularizers': 0,
        'regularization_strength': 0.0
    }
    
    total_loss = 0
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if use_update_plan:
                
                print(f"🚀 [CLIENT FIT CALLED] Path 1 here!")

                # Get current update plan for this client
                model_to_update_parts = current_update_plan["model_to_update"]
                model_as_regularizer_list = current_update_plan["model_as_regularizer"]
                
                # 1. Build the model_to_update from the main model
                model_to_update = _extract_model_parts(model, model_to_update_parts)
                
                # 2. Build the optimizer for the model_to_update
                model_to_update_optimizer = torch.optim.Adam(model_to_update.parameters(), lr=lr)
                
                # 3. Load regularizer models from shipped weights
                regularizer_models = _load_regularizer_models(model_as_regularizer_list, model, device)
                
                # Forward pass through model_to_update
                outputs_main = model_to_update(inputs)
                
                # Compute base loss
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    base_loss = criterion(outputs_main, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    base_loss = criterion(outputs_main, targets)
                
                # Determine regularizer type and compute regularized loss
                # Use configuration from REGULARIZER_PARAMS at top of file
                regularizer_type = REGULARIZER_PARAMS["type"]
                regularization_strength = REGULARIZER_PARAMS["coefficient"]
                
                # Store regularizer metadata
                regularizer_metrics['regularizer_type'] = regularizer_type
                regularizer_metrics['num_regularizers'] = len(model_as_regularizer_list)
                regularizer_metrics['regularization_strength'] = regularization_strength
                
                if regularizer_type == "weights":
                    # 3.1, 3.2, 3.3: Compute weight distance regularization
                    weight_regularizer = _compute_weight_regularizer(model_to_update, regularizer_models)
                    loss = base_loss + regularization_strength * weight_regularizer
                    
                    # Store regularizer distance for this batch
                    regularizer_metrics['regularizer_distances'].append({
                        'batch_regularizer_distance': float(weight_regularizer.item()),
                        'base_loss': float(base_loss.item()),
                        'total_loss': float(loss.item())
                    })
                    
                elif regularizer_type == "consistency loss":
                    # 4.1, 4.2, 4.3: Compute output consistency regularization
                    consistency_regularizer = _compute_consistency_regularizer(
                        model_to_update, regularizer_models, inputs, device
                    )
                    loss = base_loss + regularization_strength * consistency_regularizer
                    
                    # Store regularizer distance for this batch
                    regularizer_metrics['regularizer_distances'].append({
                        'batch_regularizer_distance': float(consistency_regularizer.item()),
                        'base_loss': float(base_loss.item()),
                        'total_loss': float(loss.item())
                    })
                
                else:
                    # Fallback to base loss
                    loss = base_loss
                
                # Log training details to client-specific log file
                log_msg = f"Round {round_num}: Using {regularizer_type} regularization with {len(model_as_regularizer_list)} regularizers, loss={loss.item():.4f}"
                if client_logger:
                    client_logger.info(log_msg)
                else:
                    print(f"[Client {client_id}] {log_msg}")  # Fallback to print if no logger
                
                # Use the specialized optimizer for model_to_update
                model_to_update_optimizer.zero_grad()
                loss.backward()
                model_to_update_optimizer.step()
                
                # Update the main model with the trained model_to_update parts
                _update_main_model_from_parts(model, model_to_update, model_to_update_parts)
                
            else:
                # Standard training without update plan
                outputs = model(inputs)
                
                # this is path 2, please output a flag
                print(f"🚀 [CLIENT FIT CALLED] Path 2 here!")

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    loss = criterion(outputs, targets)

                # add a log message
                log_msg = f"Round {round_num}: Standard training, loss={loss.item():.4f}"
                if client_logger:
                    client_logger.info(log_msg)
                else:
                    print(f"[Client {client_id}] {log_msg}")  # Fallback to print if no logger

                # Standard training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Return loss and regularizer metrics if available
    if use_update_plan and regularizer_metrics['regularizer_distances']:
        # Compute summary statistics for regularizer distances
        distances = [d['batch_regularizer_distance'] for d in regularizer_metrics['regularizer_distances']]
        regularizer_metrics['avg_regularizer_distance'] = float(np.mean(distances))
        regularizer_metrics['max_regularizer_distance'] = float(np.max(distances))
        regularizer_metrics['min_regularizer_distance'] = float(np.min(distances))
        regularizer_metrics['std_regularizer_distance'] = float(np.std(distances))
        
        return avg_loss, regularizer_metrics
    else:
        return avg_loss


def test(model, data_loader, device, data_flag):
    """Validate the model on the test set."""
    info = INFO[data_flag]
    task = info['task']
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    model.to(device)
    model.eval()
    total_loss = 0
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                targets_squeezed = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets_squeezed)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
                targets = targets.float().resize_(len(targets), 1)

            total_loss += loss.item()
            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets), 0)

    y_score = y_score.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    # calculate auc and acc
    if task == 'multi-label, binary-class':
        auc = roc_auc_score(y_true, y_score)
        # For multi-label, we need to threshold the predictions
        y_pred = (y_score > 0.5).astype(int)
        # Calculate accuracy for each label and average
        acc = np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    elif task == 'binary-class':
        auc = roc_auc_score(y_true, y_score[:, 1])  # Use probability of positive class
        acc = accuracy_score(y_true, y_score.argmax(axis=1))
    else:  # multi-class
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        acc = accuracy_score(y_true, y_score.argmax(axis=1))

    avg_loss = total_loss / len(data_loader)
    return [avg_loss, auc, acc]


def _extract_model_parts(model, model_to_update_parts):
    """
    Extract specific model parts to create model_to_update.
    
    Args:
        model: The main model
        model_to_update_parts: Dict mapping layer names to FBD block IDs
        
    Returns:
        nn.Module: A model containing only the specified parts
    """
    # Create a new model with the same architecture
    # For now, we'll create a copy and only keep the specified parts active
    import copy
    model_to_update = copy.deepcopy(model)
    
    # Note: For a more sophisticated implementation, you could create a custom
    # model that only contains the specified layers, but for now we use the full model
    # and rely on the optimizer to only update the relevant parts
    
    return model_to_update


def _load_regularizer_models(model_as_regularizer_list, template_model, device):
    """
    Load regularizer models from shipped weights.
    
    Args:
        model_as_regularizer_list: List of regularizer specifications
        template_model: Template model to use for creating regularizer models
        device: Device to load models on
        
    Returns:
        List[nn.Module]: List of regularizer models
    """
    regularizer_models = []
    
    for regularizer_spec in model_as_regularizer_list:
        # Create a copy of the template model for this regularizer
        import copy
        regularizer_model = copy.deepcopy(template_model)
        regularizer_model.to(device)
        regularizer_model.eval()  # Set to eval mode for regularization
        
        # In a full implementation, you would load the specific weights
        # from the FBD warehouse based on the regularizer_spec
        # For now, we use the current model as a placeholder
        
        regularizer_models.append(regularizer_model)
    
    return regularizer_models


def _compute_weight_regularizer(model_to_update, regularizer_models):
    """
    Compute weight distance regularization using configured distance type.
    
    Args:
        model_to_update: The model being updated
        regularizer_models: List of regularizer models
        
    Returns:
        torch.Tensor: Weight regularization loss
    """
    weight_regularizer = 0.0
    distance_type = REGULARIZER_PARAMS["distance_type"]  # No default - will fail if not set
    
    for regularizer_model in regularizer_models:
        # Compute distance between corresponding parameters
        for (name1, param1), (name2, param2) in zip(
            model_to_update.named_parameters(), 
            regularizer_model.named_parameters()
        ):
            if name1 == name2:  # Ensure we're comparing the same parameters
                param_diff = param1 - param2.detach()
                
                if distance_type.upper() == "L1":
                    # L1 (Manhattan) distance
                    weight_regularizer += torch.norm(param_diff, p=1)
                elif distance_type.upper() == "L2":
                    # L2 (Euclidean) distance
                    weight_regularizer += torch.norm(param_diff, p=2) ** 2
                elif distance_type.upper() == "COSINE":
                    # Cosine distance (1 - cosine similarity)
                    param1_flat = param1.flatten()
                    param2_flat = param2.detach().flatten()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        param1_flat.unsqueeze(0), param2_flat.unsqueeze(0)
                    )
                    weight_regularizer += 1 - cosine_sim
                elif distance_type.upper() == "KL":
                    # KL divergence (for probability distributions)
                    # Apply softmax to make parameters probability-like
                    param1_prob = torch.nn.functional.softmax(param1.flatten(), dim=0)
                    param2_prob = torch.nn.functional.softmax(param2.detach().flatten(), dim=0)
                    weight_regularizer += torch.nn.functional.kl_div(
                        param1_prob.log(), param2_prob, reduction='sum'
                    )
                else:
                    # Fail explicitly for unknown distance types
                    raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: L1, L2, COSINE, KL")
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        weight_regularizer = weight_regularizer / len(regularizer_models)
    
    return weight_regularizer


def _compute_consistency_regularizer(model_to_update, regularizer_models, inputs, device):
    """
    Compute output consistency regularization using configured distance type.
    
    Args:
        model_to_update: The model being updated
        regularizer_models: List of regularizer models
        inputs: Input batch
        device: Device for computation
        
    Returns:
        torch.Tensor: Consistency regularization loss
    """
    consistency_regularizer = 0.0
    distance_type = REGULARIZER_PARAMS["distance_type"]  # No default - will fail if not set
    
    # Get outputs from model_to_update
    model_to_update_outputs = model_to_update(inputs)
    
    for regularizer_model in regularizer_models:
        # Get outputs from regularizer model
        with torch.no_grad():  # Don't compute gradients for regularizer
            regularizer_outputs = regularizer_model(inputs)
        
        # Compute distance between outputs using configured distance type
        if distance_type.upper() == "L1":
            # L1 (Manhattan) distance
            consistency_loss = torch.nn.functional.l1_loss(
                model_to_update_outputs, regularizer_outputs.detach()
            )
        elif distance_type.upper() == "L2":
            # L2 (Euclidean) distance - MSE
            consistency_loss = torch.nn.functional.mse_loss(
                model_to_update_outputs, regularizer_outputs.detach()
            )
        elif distance_type.upper() == "COSINE":
            # Cosine distance (1 - cosine similarity)
            # Flatten outputs for cosine similarity computation
            outputs1_flat = model_to_update_outputs.flatten(start_dim=1)
            outputs2_flat = regularizer_outputs.detach().flatten(start_dim=1)
            cosine_sim = torch.nn.functional.cosine_similarity(outputs1_flat, outputs2_flat, dim=1)
            consistency_loss = (1 - cosine_sim).mean()
        elif distance_type.upper() == "KL":
            # KL divergence (for probability distributions)
            # Apply softmax to make outputs probability-like
            outputs1_prob = torch.nn.functional.softmax(model_to_update_outputs, dim=-1)
            outputs2_prob = torch.nn.functional.softmax(regularizer_outputs.detach(), dim=-1)
            consistency_loss = torch.nn.functional.kl_div(
                outputs1_prob.log(), outputs2_prob, reduction='batchmean'
            )
        elif distance_type.upper() == "JS":
            # Jensen-Shannon divergence (symmetric version of KL)
            outputs1_prob = torch.nn.functional.softmax(model_to_update_outputs, dim=-1)
            outputs2_prob = torch.nn.functional.softmax(regularizer_outputs.detach(), dim=-1)
            # Compute average distribution
            avg_prob = 0.5 * (outputs1_prob + outputs2_prob)
            # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl1 = torch.nn.functional.kl_div(avg_prob.log(), outputs1_prob, reduction='batchmean')
            kl2 = torch.nn.functional.kl_div(avg_prob.log(), outputs2_prob, reduction='batchmean')
            consistency_loss = 0.5 * (kl1 + kl2)
        else:
            # Fail explicitly for unknown distance types
            raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: L1, L2, COSINE, KL, JS")
        
        consistency_regularizer += consistency_loss
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        consistency_regularizer = consistency_regularizer / len(regularizer_models)
    
    return consistency_regularizer


def _update_main_model_from_parts(main_model, model_to_update, model_to_update_parts):
    """
    Update the main model with trained parts from model_to_update.
    
    Args:
        main_model: The main model to update
        model_to_update: The trained model parts
        model_to_update_parts: Dict mapping layer names to FBD block IDs
    """
    # Update only the specific model parts that were trained for this FBD block
    # This creates true block-level weight independence
    
    updated_state = model_to_update.state_dict()
    current_state = main_model.state_dict()
    
    # Update only the layers specified in model_to_update_parts
    for layer_name in model_to_update_parts.keys():
        if hasattr(main_model, layer_name):
            # Copy weights for this specific layer from trained model
            layer_prefix = layer_name + "."
            for param_name, param_tensor in updated_state.items():
                if param_name.startswith(layer_prefix):
                    current_state[param_name] = param_tensor.clone()
    
    # Load the updated state back to the main model
    main_model.load_state_dict(current_state)


# ====================================================================================
# FLOWER CLIENT CLASS (moved from fbd_main.py)
# ====================================================================================

class FBDFlowerClient(fl.client.Client):
    """FBD-enabled Flower client that integrates with FBD warehouse system."""
    
    def __init__(self, cid, model, train_loader, val_loader, test_loader, data_flag, device, 
                 fbd_config_path, communication_dir, client_palette, architecture='resnet18', output_dir=None):
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
        self.architecture = architecture
        self.output_dir = output_dir
        
        # Initialize client-specific logger
        self.client_logger = self._setup_client_logger()
        
        # Initialize FBD communication
        self.communication = WeightTransfer(communication_dir)
        
        # Load FBD settings
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        
        # Load update plan for determining which blocks to send back
        self.update_plan = self._load_update_plan()
        
        logging.info(f"[FBD Client {cid}] Initialized with {len(client_palette)} FBD blocks")
        self.client_logger.info(f"FBD Client {cid} initialized with {len(client_palette)} FBD blocks")
        if self.update_plan:
            logging.info(f"[FBD Client {cid}] Loaded update plan for {len(self.update_plan)} rounds")

    def _setup_client_logger(self):
        """Set up client-specific logger that writes to a file."""
        # Create client-specific logger
        client_logger = logging.getLogger(f"FBDClient_{self.cid}")
        client_logger.setLevel(logging.INFO)
        
        # Avoid adding handlers multiple times
        if not client_logger.handlers:
            # Create log file path
            if self.output_dir:
                log_file = os.path.join(self.output_dir, f"client_{self.cid}_training.log")
                os.makedirs(self.output_dir, exist_ok=True)
            else:
                # Fallback to current directory if no output_dir provided
                log_file = f"client_{self.cid}_training.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            client_logger.addHandler(file_handler)
            
            # Prevent propagation to root logger to avoid duplicate logging
            client_logger.propagate = False
        
        return client_logger

    def _load_update_plan(self):
        """Load the update plan to determine which blocks this client should update each round."""
        try:
            # Try multiple possible locations for update plan
            update_plan_paths = [
                "fbd_record/update_plan.json",
                "update_plan.json",
                os.path.join(self.communication_dir, "update_plan.json")
            ]
            
            for plan_path in update_plan_paths:
                if os.path.exists(plan_path):
                    with open(plan_path, 'r') as f:
                        update_plan = json.load(f)
                    logging.info(f"[FBD Client {self.cid}] Loaded update plan from {plan_path}")
                    return update_plan
            
            logging.warning(f"[FBD Client {self.cid}] No update plan found - will send all palette blocks")
            return None
            
        except Exception as e:
            logging.warning(f"[FBD Client {self.cid}] Failed to load update plan: {e}")
            return None
    
    def _get_updated_blocks_for_round(self, round_num):
        """
        Get the list of block IDs that this client should send back for the given round.
        Only blocks in 'model_to_update' should be sent, not regularizer blocks.
        
        Args:
            round_num (int): Current round number
            
        Returns:
            list: List of block IDs this client should send back
        """
        if not self.update_plan:
            # Fallback: send all palette blocks if no update plan
            logging.warning(f"[FBD Client {self.cid}] No update plan - sending all palette blocks")
            return list(self.client_palette.keys())
        
        round_str = str(round_num)
        client_str = str(self.cid)
        
        # Check if this round and client exist in update plan
        if round_str not in self.update_plan:
            logging.warning(f"[FBD Client {self.cid}] Round {round_num} not found in update plan")
            return []
        
        if client_str not in self.update_plan[round_str]:
            logging.warning(f"[FBD Client {self.cid}] Client {self.cid} not found in round {round_num} update plan")
            return []
        
        # Extract the blocks this client should update (not regularize)
        client_plan = self.update_plan[round_str][client_str]
        model_to_update = client_plan.get("model_to_update", {})
        
        # Get the block IDs from model_to_update values
        blocks_to_update = list(model_to_update.values())
        
        logging.info(f"[FBD Client {self.cid}] Round {round_num}: Should update blocks {blocks_to_update}")
        self.client_logger.info(f"Round {round_num}: Update plan specifies updating blocks {blocks_to_update}")
        
        return blocks_to_update

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
        
        # FBD: Send updated weights to warehouse (ONLY send blocks that were actually updated)
        # Use update plan to determine which blocks this client updated in this round
        blocks_to_send = self._get_updated_blocks_for_round(round_num)
        
        if blocks_to_send:
            extracted_weights = {}
            for block_id in blocks_to_send:
                if block_id in self.client_palette:
                    model_part = self.client_palette[block_id]['model_part']
                    # Extract weights for this model part
                    part_weights = self.model.send_for_dict([model_part])
                    if part_weights:
                        extracted_weights[block_id] = part_weights[model_part]
            
            # Send only the blocks that were actually updated to warehouse
            if extracted_weights:
                self.communication.client_send_weights(self.cid, round_num, extracted_weights, list(extracted_weights.keys()))
                logging.info(f"[FBD Client {self.cid}] Sent {len(extracted_weights)} UPDATED FBD blocks to server (from {len(blocks_to_send)} planned updates)")
                self.client_logger.info(f"Round {round_num}: Sent updated blocks {list(extracted_weights.keys())} to server")
            else:
                logging.warning(f"[FBD Client {self.cid}] No weights extracted for planned update blocks: {blocks_to_send}")
        else:
            logging.warning(f"[FBD Client {self.cid}] No blocks to update in round {round_num} according to update plan")
        
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
        return train(self.model, self.train_loader, epochs=1, device=self.device, 
                    data_flag=self.data_flag, lr=lr, current_update_plan=current_update_plan,
                    client_id=self.cid, round_num=round_num, client_logger=self.client_logger)

    def _test_model(self, data_loader):
        """Test the model."""
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



 