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
    
    # Debug: Output path decision logic
    print(f"[Path Decision Debug] Client {client_id}, Round {round_num}")
    print(f"  - current_update_plan is not None: {current_update_plan is not None}")
    print(f"  - client_id is not None: {client_id is not None}")
    print(f"  - round_num is not None: {round_num is not None}")
    print(f"  - Final use_update_plan: {use_update_plan}")
    if current_update_plan is not None:
        print(f"  - Update plan keys: {list(current_update_plan.keys())}")
        if "model_to_update" in current_update_plan:
            print(f"  - model_to_update: {current_update_plan['model_to_update']}")
        if "model_as_regularizer" in current_update_plan:
            print(f"  - num_regularizers: {len(current_update_plan['model_as_regularizer'])}")
    else:
        print(f"  - Update plan is None - no plan received from server")
    
    # Initialize regularizer metrics tracking
    regularizer_metrics = {
        'regularizer_distances': [],
        'regularizer_type': None,
        'num_regularizers': 0,
        'regularization_strength': 0.0
    }
    
    # ========== MOVE MODEL BUILDING OUTSIDE THE LOOP ==========
    if use_update_plan:
        print(f"[Client {client_id}] Round {round_num}: Entering Training Path 1 (FBD Regularized Training)")
        print(f"[Model Building] Building models and optimizers for Path 1...")
        
        # Get current update plan for this client
        model_to_update_parts = current_update_plan["model_to_update"]
        model_as_regularizer_list = current_update_plan["model_as_regularizer"]
        
        print(f"[Model Building] Update plan details:")
        print(f"  - model_to_update_parts: {model_to_update_parts}")
        print(f"  - model_as_regularizer_list: {model_as_regularizer_list}")
        
        # 1. Build the model_to_update from the main model (ONCE per round)
        print(f"[Model Building] Step 1: Extracting model parts...")
        model_to_update = _extract_model_parts(model, model_to_update_parts)
        print(f"[Model Building] ✅ model_to_update created with {sum(p.numel() for p in model_to_update.parameters())} parameters")
        
        # 2. Build the optimizer for the model_to_update (ONCE per round)
        print(f"[Model Building] Step 2: Creating optimizer...")
        model_to_update_optimizer = torch.optim.Adam(model_to_update.parameters(), lr=lr)
        print(f"[Model Building] ✅ Optimizer created for model_to_update")
        
        # 3. Load regularizer models from shipped weights (ONCE per round)
        print(f"[Model Building] Step 3: Loading regularizer models...")
        regularizer_models = _load_regularizer_models(model_as_regularizer_list, model, device)
        print(f"[Model Building] ✅ Loaded {len(regularizer_models)} regularizer models")
        
        # Get regularizer configuration (ONCE per round)
        regularizer_type = REGULARIZER_PARAMS["type"]
        regularization_strength = REGULARIZER_PARAMS["coefficient"]
        
        # Store regularizer metadata
        regularizer_metrics['regularizer_type'] = regularizer_type
        regularizer_metrics['num_regularizers'] = len(model_as_regularizer_list)
        regularizer_metrics['regularization_strength'] = regularization_strength
        
        print(f"[Model Building] ✅ Built {len(regularizer_models)} regularizer models")
        print(f"[Model Building] Using {regularizer_type} regularization with strength {regularization_strength}")
    else:
        print(f"[Client {client_id}] Round {round_num}: Will use Training Path 2 (Standard Training)")
    
    # ========== TRAINING LOOP ==========
    total_loss = 0
    batch_count = 0
    for epoch in range(epochs):
        print(f"[Training Loop] Client {client_id} Round {round_num}: Starting epoch {epoch+1}/{epochs}")
        for inputs, targets in train_loader:
            batch_count += 1
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_update_plan:
                # Training Path 1 - FBD Regularized Training
                # Models are already built above, just do forward/backward pass
                
                if batch_count == 1:  # Log details for first batch only
                    print(f"[Training Path 1] Client {client_id} Round {round_num}: Processing batch {batch_count} with FBD regularization")
                    print(f"[Training Path 1] Input shape: {inputs.shape}, Target shape: {targets.shape}")
                
                # Forward pass through model_to_update
                outputs_main = model_to_update(inputs)
                
                # Compute base loss
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    base_loss = criterion(outputs_main, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    base_loss = criterion(outputs_main, targets)
                
                # Compute regularized loss using pre-built models
                if regularizer_type == "weights":
                    # 3.1, 3.2, 3.3: Compute weight distance regularization
                    weight_regularizer = _compute_weight_regularizer(model_to_update, regularizer_models)
                    loss = base_loss + regularization_strength * weight_regularizer
                    
                    if batch_count == 1:  # Log details for first batch only
                        print(f"[Training Path 1] Batch {batch_count}: base_loss={base_loss.item():.6f}, weight_reg={weight_regularizer.item():.6f}, total_loss={loss.item():.6f}")
                    
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
                    
                    if batch_count == 1:  # Log details for first batch only
                        print(f"[Training Path 1] Batch {batch_count}: base_loss={base_loss.item():.6f}, consistency_reg={consistency_regularizer.item():.6f}, total_loss={loss.item():.6f}")
                    
                    # Store regularizer distance for this batch
                    regularizer_metrics['regularizer_distances'].append({
                        'batch_regularizer_distance': float(consistency_regularizer.item()),
                        'base_loss': float(base_loss.item()),
                        'total_loss': float(loss.item())
                    })
                
                else:
                    # Fallback to base loss
                    loss = base_loss
                    if batch_count == 1:
                        print(f"[Training Path 1] Batch {batch_count}: No regularization applied, loss={loss.item():.6f}")
                
                # Use the pre-built optimizer for model_to_update
                model_to_update_optimizer.zero_grad()
                loss.backward()
                model_to_update_optimizer.step()
                
                # Update the main model with the trained model_to_update parts
                _update_main_model_from_parts(model, model_to_update, model_to_update_parts)
                
            else:
                # Training Path 2 - Standard Training
                if batch_count == 1:  # Log details for first batch only
                    print(f"[Training Path 2] Client {client_id} Round {round_num}: Processing batch {batch_count} with standard training")
                    print(f"[Update Strategy] Client {client_id} Round {round_num}: Using standard FL training strategy")
                    print(f"[Update Strategy] Reason for Path 2: update_plan_received={current_update_plan is not None}, client_id_provided={client_id is not None}, round_num_provided={round_num is not None}")
                    print(f"[Update Strategy] Training mode: Standard federated learning without FBD regularization")
                    print(f"[Update Strategy] Optimizer: Adam with lr={lr}")
                    print(f"[Update Strategy] Model architecture: Full model training (no part-based updates)")

                # Standard training without update plan
                outputs = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    loss = criterion(outputs, targets)

                if batch_count == 1:  # Log details for first batch only
                    print(f"[Training Path 2] Batch {batch_count}: Standard training loss={loss.item():.6f}")

                # Enhanced log message for Path 2
                log_msg = f"Round {round_num}: [Update Strategy] Standard training, loss={loss.item():.4f}, task={task}"
                if client_logger:
                    client_logger.info(log_msg)
                else:
                    print(f"[Client {client_id}] {log_msg}")  # Fallback to print if no logger

                print(f"[Update Strategy] Client {client_id} Round {round_num}: Batch loss={loss.item():.6f}, task_type={task}")

                # Standard training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        print(f"[Training Loop] Client {client_id} Round {round_num}: Completed epoch {epoch+1}/{epochs}, avg_loss_so_far={total_loss/batch_count:.6f}")

    # Log training summary after all epochs
    if use_update_plan:
        log_msg = f"Round {round_num}: FBD training completed - {regularizer_type} regularization with {len(model_as_regularizer_list)} regularizers, avg_loss={total_loss/len(train_loader):.4f}"
        if client_logger:
            client_logger.info(log_msg)
        else:
            print(f"[Client {client_id}] {log_msg}")
        print(f"[Training Summary] Client {client_id} Round {round_num}: Used Path 1 (FBD) - processed {batch_count} batches")
    else:
        print(f"[Training Summary] Client {client_id} Round {round_num}: Used Path 2 (Standard) - processed {batch_count} batches")

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
    print(f"[_extract_model_parts] Input model_to_update_parts: {model_to_update_parts}")
    print(f"[_extract_model_parts] Main model type: {type(model)}")
    print(f"[_extract_model_parts] Main model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create a new model with the same architecture
    # For now, we'll create a copy and only keep the specified parts active
    import copy
    model_to_update = copy.deepcopy(model)
    
    print(f"[_extract_model_parts] ✅ Created deep copy of model")
    print(f"[_extract_model_parts] model_to_update parameters: {sum(p.numel() for p in model_to_update.parameters())}")
    
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
    print(f"[_load_regularizer_models] Loading {len(model_as_regularizer_list)} regularizer models")
    print(f"[_load_regularizer_models] Regularizer specs: {model_as_regularizer_list}")
    print(f"[_load_regularizer_models] Template model type: {type(template_model)}")
    print(f"[_load_regularizer_models] Device: {device}")
    
    regularizer_models = []
    
    for i, regularizer_spec in enumerate(model_as_regularizer_list):
        print(f"[_load_regularizer_models] Creating regularizer {i+1}/{len(model_as_regularizer_list)}: {regularizer_spec}")
        
        # Create a copy of the template model for this regularizer
        import copy
        regularizer_model = copy.deepcopy(template_model)
        regularizer_model.to(device)
        regularizer_model.eval()  # Set to eval mode for regularization
        
        print(f"[_load_regularizer_models] ✅ Created regularizer model {i+1} with {sum(p.numel() for p in regularizer_model.parameters())} parameters")
        
        # In a full implementation, you would load the specific weights
        # from the FBD warehouse based on the regularizer_spec
        # For now, we use the current model as a placeholder
        
        regularizer_models.append(regularizer_model)
    
    print(f"[_load_regularizer_models] ✅ Loaded {len(regularizer_models)} regularizer models")
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
    
    print(f"[_compute_weight_regularizer] Computing {distance_type} distance with {len(regularizer_models)} regularizer models")
    
    for reg_idx, regularizer_model in enumerate(regularizer_models):
        reg_loss_for_this_model = 0.0
        param_count = 0
        
        # Compute distance between corresponding parameters
        for (name1, param1), (name2, param2) in zip(
            model_to_update.named_parameters(), 
            regularizer_model.named_parameters()
        ):
            if name1 == name2:  # Ensure we're comparing the same parameters
                param_count += 1
                param_diff = param1 - param2.detach()
                
                if distance_type.upper() == "L1":
                    # L1 (Manhattan) distance
                    param_loss = torch.norm(param_diff, p=1)
                elif distance_type.upper() == "L2":
                    # L2 (Euclidean) distance
                    param_loss = torch.norm(param_diff, p=2) ** 2
                elif distance_type.upper() == "COSINE":
                    # Cosine distance (1 - cosine similarity)
                    param1_flat = param1.flatten()
                    param2_flat = param2.detach().flatten()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        param1_flat.unsqueeze(0), param2_flat.unsqueeze(0)
                    )
                    param_loss = 1 - cosine_sim
                elif distance_type.upper() == "KL":
                    # KL divergence (for probability distributions)
                    # Apply softmax to make parameters probability-like
                    param1_prob = torch.nn.functional.softmax(param1.flatten(), dim=0)
                    param2_prob = torch.nn.functional.softmax(param2.detach().flatten(), dim=0)
                    param_loss = torch.nn.functional.kl_div(
                        param1_prob.log(), param2_prob, reduction='sum'
                    )
                else:
                    # Fail explicitly for unknown distance types
                    raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: L1, L2, COSINE, KL")
                
                reg_loss_for_this_model += param_loss
        
        weight_regularizer += reg_loss_for_this_model
        print(f"[_compute_weight_regularizer] Regularizer {reg_idx+1}: loss={reg_loss_for_this_model.item():.6f} across {param_count} parameter groups")
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        weight_regularizer = weight_regularizer / len(regularizer_models)
    
    print(f"[_compute_weight_regularizer] ✅ Final weight regularizer: {weight_regularizer.item():.6f}")
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
    
    print(f"[_compute_consistency_regularizer] Computing {distance_type} consistency with {len(regularizer_models)} regularizer models")
    print(f"[_compute_consistency_regularizer] Input shape: {inputs.shape}")
    
    # Get outputs from model_to_update
    model_to_update_outputs = model_to_update(inputs)
    print(f"[_compute_consistency_regularizer] model_to_update output shape: {model_to_update_outputs.shape}")
    
    for reg_idx, regularizer_model in enumerate(regularizer_models):
        # Get outputs from regularizer model
        with torch.no_grad():  # Don't compute gradients for regularizer
            regularizer_outputs = regularizer_model(inputs)
        
        print(f"[_compute_consistency_regularizer] Regularizer {reg_idx+1} output shape: {regularizer_outputs.shape}")
        
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
        print(f"[_compute_consistency_regularizer] Regularizer {reg_idx+1}: consistency_loss={consistency_loss.item():.6f}")
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        consistency_regularizer = consistency_regularizer / len(regularizer_models)
    
    print(f"[_compute_consistency_regularizer] ✅ Final consistency regularizer: {consistency_regularizer.item():.6f}")
    return consistency_regularizer


def _update_main_model_from_parts(main_model, model_to_update, model_to_update_parts):
    """
    Update the main model with trained parts from model_to_update.
    
    Args:
        main_model: The main model to update
        model_to_update: The trained model parts
        model_to_update_parts: Dict mapping layer names to FBD block IDs
    """
    print(f"[_update_main_model_from_parts] Updating main model from trained parts")
    print(f"[_update_main_model_from_parts] model_to_update_parts: {model_to_update_parts}")
    
    # Update only the specific model parts that were trained for this FBD block
    # This creates true block-level weight independence
    
    updated_state = model_to_update.state_dict()
    current_state = main_model.state_dict()
    
    print(f"[_update_main_model_from_parts] Updated state has {len(updated_state)} parameters")
    print(f"[_update_main_model_from_parts] Current state has {len(current_state)} parameters")
    
    updated_layers = []
    
    # Update only the layers specified in model_to_update_parts
    for layer_name in model_to_update_parts.keys():
        if hasattr(main_model, layer_name):
            # Copy weights for this specific layer from trained model
            layer_prefix = layer_name + "."
            layer_params_updated = 0
            for param_name, param_tensor in updated_state.items():
                if param_name.startswith(layer_prefix):
                    current_state[param_name] = param_tensor.clone()
                    layer_params_updated += 1
            updated_layers.append(f"{layer_name}({layer_params_updated})")
            print(f"[_update_main_model_from_parts] Updated layer {layer_name}: {layer_params_updated} parameters")
        else:
            print(f"[_update_main_model_from_parts] ❌ Layer {layer_name} not found in main model")
    
    # Load the updated state back to the main model
    main_model.load_state_dict(current_state)
    print(f"[_update_main_model_from_parts] ✅ Updated main model with layers: {updated_layers}")
    
    return updated_layers


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
        
        # Debug: Check if update plan was received from server
        current_update_plan = config.get("current_update_plan", None)
        print(f"[Update Plan Debug] Client {self.cid} Round {round_num}:")
        print(f"  - Config keys received: {list(config.keys())}")
        print(f"  - current_update_plan in config: {'current_update_plan' in config}")
        print(f"  - current_update_plan is not None: {current_update_plan is not None}")
        if current_update_plan is not None:
            print(f"  - Update plan structure: {list(current_update_plan.keys())}")
            print(f"  - CLIENT {self.cid} RECEIVED UPDATE PLAN from server!")
            print(f"  - model_to_update: {current_update_plan.get('model_to_update', 'Not found')}")
            print(f"  - model_as_regularizer: {current_update_plan.get('model_as_regularizer', 'Not found')}")
        else:
            print(f"  - CLIENT {self.cid} DID NOT RECEIVE UPDATE PLAN from server")
            print(f"  - This will lead to Path 2 (Standard Training)")
        
        # Store model weights BEFORE training for comparison
        model_weights_before = {}
        for name, param in self.model.named_parameters():
            model_weights_before[name] = param.data.clone()
        print(f"[Weight Tracking] Client {self.cid} Round {round_num}: Stored {len(model_weights_before)} parameter tensors before training")
        
        # FBD: Receive weights from server (shipping phase)
        try:
            received_weights = self.communication.client_receive_weights(self.cid, round_num)
            if received_weights:
                # print(f"[FBD Shipping] Client {self.cid} Round {round_num}: Received {len(received_weights)} model parts from warehouse")
                # print(f"[FBD Shipping] Received weight keys: {list(received_weights.keys())}")
                self.model.load_from_dict(received_weights)
                # print(f"[FBD Shipping] ✅ Successfully loaded received weights into model")
            # else:
                # print(f"[FBD Shipping] Client {self.cid} Round {round_num}: No weights received from warehouse - using current model")
        except (TimeoutError, FileNotFoundError) as e:
            # print(f"[FBD Shipping] Client {self.cid} Round {round_num}: Exception during weight receiving: {e}")
            # print(f"[FBD Shipping] Using current model weights")
            pass
        
        # Perform local training
        train_result = self._train_model(local_lr, current_update_plan=current_update_plan, round_num=round_num)
        
        # Handle different return types from train function
        if isinstance(train_result, tuple):
            train_loss, regularizer_metrics = train_result
        else:
            train_loss = train_result
            regularizer_metrics = None
        
        # Check if model weights actually changed during training
        model_weights_after = {}
        total_weight_change = 0.0
        changed_layers = []
        for name, param in self.model.named_parameters():
            model_weights_after[name] = param.data.clone()
            weight_diff = torch.norm(model_weights_after[name] - model_weights_before[name]).item()
            total_weight_change += weight_diff
            if weight_diff > 1e-8:  # Threshold for detecting change
                changed_layers.append(f"{name}:{weight_diff:.6f}")
        
        print(f"[Weight Tracking] Client {self.cid} Round {round_num}: Total weight change = {total_weight_change:.6f}")
        print(f"[Weight Tracking] Changed layers ({len(changed_layers)}): {changed_layers[:5]}...")  # Show first 5
        
        if total_weight_change < 1e-6:
            print(f"[Weight Tracking] ⚠️  WARNING: Model weights barely changed during training!")
        else:
            print(f"[Weight Tracking] ✅ Model weights were updated during training")
        
        # Evaluate after training
        train_loss, train_auc, train_acc = self._test_model(self.train_loader)
        val_loss, val_auc, val_acc = self._test_model(self.val_loader)
        
        logging.info(f"[FBD Client {self.cid}] Round {round_num}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # FBD: Send updated weights to warehouse (ONLY send blocks that were actually updated)
        # Use update plan to determine which blocks this client updated in this round
        blocks_to_send = self._get_updated_blocks_for_round(round_num)
        # print(f"[FBD Update] Client {self.cid} Round {round_num}: Planning to send {len(blocks_to_send)} blocks: {blocks_to_send}")
        
        if blocks_to_send:
            extracted_weights = {}
            for block_id in blocks_to_send:
                if block_id in self.client_palette:
                    model_part = self.client_palette[block_id]['model_part']
                    # print(f"[FBD Update] Extracting weights for block {block_id} (model_part: {model_part})")
                    # Extract weights for this model part
                    part_weights = self.model.send_for_dict([model_part])
                    if part_weights:
                        extracted_weights[block_id] = part_weights[model_part]
                        # print(f"[FBD Update] ✅ Extracted weights for block {block_id}: {len(part_weights[model_part])} parameters")
                    # else:
                        # print(f"[FBD Update] ❌ Failed to extract weights for block {block_id}")
                # else:
                    # print(f"[FBD Update] ❌ Block {block_id} not found in client palette")
            
            # Send only the blocks that were actually updated to warehouse
            if extracted_weights:
                # print(f"[FBD Update] Client {self.cid} Round {round_num}: Sending {len(extracted_weights)} blocks to warehouse...")
                self.communication.client_send_weights(self.cid, round_num, extracted_weights, list(extracted_weights.keys()))
                # print(f"[FBD Update] ✅ Successfully sent {len(extracted_weights)} UPDATED FBD blocks to warehouse")
                # print(f"[FBD Update] Sent block IDs: {list(extracted_weights.keys())}")
                logging.info(f"[FBD Client {self.cid}] Sent {len(extracted_weights)} UPDATED FBD blocks to server (from {len(blocks_to_send)} planned updates)")
                self.client_logger.info(f"Round {round_num}: Sent updated blocks {list(extracted_weights.keys())} to server")
            # else:
                # print(f"[FBD Update] ❌ No weights extracted for planned update blocks: {blocks_to_send}")
                # logging.warning(f"[FBD Client {self.cid}] No weights extracted for planned update blocks: {blocks_to_send}")
        # else:
            # print(f"[FBD Update] ❌ No blocks to update in round {round_num} according to update plan")
            # logging.warning(f"[FBD Client {self.cid}] No blocks to update in round {round_num} according to update plan")
        
        # Note: Collection phase will be handled by server using these weights
        
        # Return metrics (no parameters needed for FBD - using file communication)
        metrics_dict = {
            "train_loss": train_loss,
            "train_auc": train_auc, 
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "total_weight_change": total_weight_change,  # Add weight change tracking
            "num_changed_layers": len(changed_layers)
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
        
        print(f"[FBD Client {self.cid}] Round {round_num} COMPLETE: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, weight_change={total_weight_change:.6f}")
        
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



 