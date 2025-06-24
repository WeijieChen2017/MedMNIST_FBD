"""
FBD Client Implementation for Federated Learning

This module contains all FBD client-related functionality consolidated from multiple files:

Classes:
    - FBDFlowerClient: Main Flower-based client for FBD federated learning (moved from fbd_main.py)
    - FBDClient: Standalone FBD client for testing and development purposes (original)

Functions:
    - train(): Model training with FBD regularization support (moved from client.py)
    - test(): Model evaluation (moved from client.py)
    - _extract_model_parts(): Helper for FBD block extraction
    - _load_regularizer_models(): Helper for loading regularizer models
    - _compute_weight_regularizer(): Weight-based regularization computation
    - _compute_consistency_regularizer(): Output consistency regularization computation
    - _update_main_model_from_parts(): Model update helper

This consolidation improves code organization by keeping all client-side FBD logic in one place.
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

def train(model, train_loader, epochs, device, data_flag, lr, current_update_plan=None, client_id=None, round_num=None):
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
                
                print(f"[Client {client_id}] Round {round_num}: Using {regularizer_type} regularization with {len(model_as_regularizer_list)} regularizers, loss={loss.item():.4f}")
                
                # Use the specialized optimizer for model_to_update
                model_to_update_optimizer.zero_grad()
                loss.backward()
                model_to_update_optimizer.step()
                
                # Update the main model with the trained model_to_update parts
                _update_main_model_from_parts(model, model_to_update, model_to_update_parts)
                
            else:
                # Standard training without update plan
                outputs = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    loss = criterion(outputs, targets)
                
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
                 fbd_config_path, communication_dir, client_palette, architecture='resnet18'):
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
        return train(self.model, self.train_loader, epochs=1, device=self.device, 
                    data_flag=self.data_flag, lr=lr, current_update_plan=current_update_plan,
                    client_id=self.cid, round_num=round_num)

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


# ====================================================================================
# ORIGINAL FBD CLIENT CLASS (for backward compatibility)
# ====================================================================================

class FBDClient:
    """
    FBD Federated Learning Client.
    Handles weight updates, local training, and communication with server.
    """
    
    def __init__(self,
                 client_id: int,
                 fbd_config_path: str,
                 communication_dir: str = "fbd_comm",
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 learning_rate: float = 0.001,
                 device: str = 'cpu',
                 norm: str = 'bn',
                 architecture: str = 'resnet18'):
        """
        Initialize FBD client.
        
        Args:
            client_id: Unique client identifier
            fbd_config_path: Path to FBD configuration file
            communication_dir: Directory for communication files
            num_classes: Number of output classes
            input_shape: Input tensor shape
            learning_rate: Learning rate for local training
            device: Device for computation
            norm: Normalization type ('bn', 'in', 'ln')
            architecture: Model architecture ('resnet18', 'resnet50')
        """
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.architecture = architecture
        self.norm = norm
        
        # Load FBD configuration
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        
        # Initialize local model
        self.model = get_fbd_model(architecture, norm, input_shape[0], num_classes)
        self.model.to(device)
        
        # Initialize trainer
        self.trainer = LocalTrainer(
            model=self.model,
            num_classes=num_classes,
            input_shape=input_shape,
            learning_rate=learning_rate,
            device=device
        )
        
        # Initialize communication
        self.comm = WeightTransfer(communication_dir)
        
        # Client state
        self.current_round = 0
        self.training_history = []
        self.client_palette = self._get_client_palette()
        
        print(f"Client {client_id} initialized with {len(self.client_palette)} accessible blocks")
    
    def _get_client_palette(self) -> Dict[str, Dict]:
        """
        Get the model palette (accessible blocks) for this client.
        
        Returns:
            Dict: Client's model palette
        """
        if self.client_id in self.fbd_info["clients"]:
            # Get the models (colors) this client has access to
            client_colors = self.fbd_info["clients"][self.client_id]
            
            # Create the model palette for this client
            model_palette = {}
            for fbd_id, fbd_entry in self.fbd_trace.items():
                if fbd_entry["color"] in client_colors:
                    palette_entry = fbd_entry.copy()
                    
                    # Remove color information if not transparent to client
                    if not self.transparent_to_client:
                        palette_entry.pop("color", None)
                    
                    model_palette[fbd_id] = palette_entry
            
            return model_palette
        else:
            # Default: client has access to all models if not specified
            default_palette = self.fbd_trace.copy()
            
            # Remove color information if not transparent to client
            if not self.transparent_to_client:
                for fbd_id, fbd_entry in default_palette.items():
                    fbd_entry.pop("color", None)
            
            return default_palette
    
    def receive_weights_from_server(self, round_num: int) -> Dict[str, Dict]:
        """
        Receive weights from server for current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dict: Received weights mapping model parts to state_dicts
        """
        try:
            weights_dict = self.comm.client_receive_weights(self.client_id, round_num)
            print(f"Client {self.client_id} received weights for round {round_num}")
            print(f"  Model parts: {list(weights_dict.keys())}")
            return weights_dict
        except Exception as e:
            print(f"Client {self.client_id} failed to receive weights: {e}")
            return {}
    
    def update_local_model(self, weights_dict: Dict[str, Dict]):
        """
        Update local model with received weights.
        
        Args:
            weights_dict: Weights mapping model parts to state_dicts
        """
        if not weights_dict:
            print(f"Client {self.client_id}: No weights to update")
            return
        
        # Use FBD model's load_from_dict method
        self.model.load_from_dict(weights_dict)
        print(f"Client {self.client_id} updated model with {len(weights_dict)} parts")
    
    def perform_local_training(self, 
                             epochs: int = 1,
                             batch_size: int = 32,
                             num_batches: int = 10,
                             verbose: bool = False) -> Dict[str, Any]:
        """
        Perform local training.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            num_batches: Number of batches per epoch
            verbose: Whether to print training progress
            
        Returns:
            Dict: Training results
        """
        if verbose:
            print(f"Client {self.client_id} starting local training...")
        
        training_results = self.trainer.train_multiple_epochs(
            epochs=epochs,
            batch_size=batch_size,
            num_batches=num_batches,
            verbose=verbose
        )
        
        # Add client info to results
        training_results['client_id'] = self.client_id
        training_results['round'] = self.current_round
        
        self.training_history.append(training_results)
        
        if verbose:
            print(f"Client {self.client_id} completed training:")
            print(f"  Final accuracy: {training_results['final_accuracy']:.2f}%")
            print(f"  Average loss: {training_results['avg_loss']:.4f}")
        
        return training_results
    
    def receive_request_list(self, round_num: int) -> List[str]:
        """
        Receive request list from server.
        
        Args:
            round_num: Current round number
            
        Returns:
            List[str]: List of block IDs to send back
        """
        try:
            request_list = self.comm.client_receive_request_list(self.client_id, round_num)
            print(f"Client {self.client_id} received request for {len(request_list)} blocks")
            return request_list
        except Exception as e:
            print(f"Client {self.client_id} failed to receive request list: {e}")
            return []
    
    def extract_requested_weights(self, request_list: List[str]) -> Dict[str, Dict]:
        """
        Extract requested weights from local model.
        
        Args:
            request_list: List of block IDs to extract
            
        Returns:
            Dict: Extracted weights mapping block IDs to state_dicts
        """
        extracted_weights = {}
        
        # Group blocks by model part
        parts_to_extract = {}
        for block_id in request_list:
            if block_id in self.fbd_trace:
                model_part = self.fbd_trace[block_id]['model_part']
                if model_part not in parts_to_extract:
                    parts_to_extract[model_part] = []
                parts_to_extract[model_part].append(block_id)
        
        # Extract weights using FBD model's send_for_dict method
        model_weights = self.model.send_for_dict(list(parts_to_extract.keys()))
        
        # Map back to block IDs
        for model_part, block_ids in parts_to_extract.items():
            if model_part in model_weights:
                for block_id in block_ids:
                    extracted_weights[block_id] = model_weights[model_part]
        
        print(f"Client {self.client_id} extracted weights for {len(extracted_weights)} blocks")
        return extracted_weights
    
    def send_weights_to_server(self, round_num: int, weights_dict: Dict[str, Dict], request_list: List[str]):
        """
        Send extracted weights back to server.
        
        Args:
            round_num: Current round number
            weights_dict: Extracted weights
            request_list: Original request list
        """
        try:
            self.comm.client_send_weights(self.client_id, round_num, weights_dict, request_list)
            print(f"Client {self.client_id} sent {len(weights_dict)} blocks to server")
        except Exception as e:
            print(f"Client {self.client_id} failed to send weights: {e}")
    
    def run_round(self, 
                  round_num: int,
                  training_epochs: int = 1,
                  batch_size: int = 32,
                  num_batches: int = 10,
                  verbose: bool = False) -> Dict[str, Any]:
        """
        Execute one complete round of federated learning.
        
        Args:
            round_num: Current round number
            training_epochs: Number of local training epochs
            batch_size: Batch size for training
            num_batches: Number of batches per epoch
            verbose: Whether to print progress
            
        Returns:
            Dict: Round results
        """
        self.current_round = round_num
        
        if verbose:
            print(f"\n=== Client {self.client_id} Round {round_num} ===")
        
        round_start_time = time.time()
        
        # Phase 1: Receive weights from server
        received_weights = self.receive_weights_from_server(round_num)
        if received_weights:
            self.update_local_model(received_weights)
        
        # Phase 2: Local training
        training_results = self.perform_local_training(
            epochs=training_epochs,
            batch_size=batch_size,
            num_batches=num_batches,
            verbose=verbose
        )
        
        # Phase 3: Receive request list and send weights back
        request_list = self.receive_request_list(round_num)
        if request_list:
            extracted_weights = self.extract_requested_weights(request_list)
            self.send_weights_to_server(round_num, extracted_weights, request_list)
        
        round_time = time.time() - round_start_time
        
        round_results = {
            'client_id': self.client_id,
            'round': round_num,
            'round_time': round_time,
            'received_parts': list(received_weights.keys()) if received_weights else [],
            'requested_blocks': request_list,
            'training_results': training_results
        }
        
        if verbose:
            print(f"Client {self.client_id} completed round {round_num} in {round_time:.2f}s")
        
        return round_results
    
    def evaluate_model(self, batch_size: int = 32, num_batches: int = 5) -> Dict[str, float]:
        """
        Evaluate current local model.
        
        Args:
            batch_size: Batch size for evaluation
            num_batches: Number of batches for evaluation
            
        Returns:
            Dict: Evaluation metrics
        """
        return self.trainer.evaluate_model(batch_size=batch_size, num_batches=num_batches)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get complete training history.
        
        Returns:
            List: Training history for all rounds
        """
        return self.training_history.copy()
    
    def save_client_state(self, filepath: str):
        """
        Save client state including model and training history.
        
        Args:
            filepath: Path to save client state
        """
        state = {
            'client_id': self.client_id,
            'current_round': self.current_round,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'trainer_stats': self.trainer.get_training_stats(),
            'client_palette': self.client_palette
        }
        
        torch.save(state, filepath)
        print(f"Client {self.client_id} state saved to {filepath}")
    
    def load_client_state(self, filepath: str):
        """
        Load client state.
        
        Args:
            filepath: Path to load client state from
        """
        state = torch.load(filepath, map_location=self.device)
        
        self.client_id = state['client_id']
        self.current_round = state['current_round']
        self.model.load_state_dict(state['model_state_dict'])
        self.training_history = state['training_history']
        self.client_palette = state['client_palette']
        
        print(f"Client {self.client_id} state loaded from {filepath}") 