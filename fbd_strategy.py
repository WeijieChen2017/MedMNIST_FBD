"""
FBD Evaluation Strategies
Provides different evaluation methods for FBD federated learning
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import roc_auc_score, accuracy_score

from models import ResNet18_FBD_BN, ResNet18_FBD_IN, ResNet18_FBD_LN


def get_resnet18_fbd_model(norm: str, in_channels: int, num_classes: int):
    """Get the appropriate ResNet18 FBD model based on normalization type."""
    if norm == 'bn':
        return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'in':
        return ResNet18_FBD_IN(in_channels=in_channels, num_classes=num_classes)
    elif norm == 'ln':
        return ResNet18_FBD_LN(in_channels=in_channels, num_classes=num_classes)
    else:
        # Default to batch normalization if norm type is not specified or unknown
        return ResNet18_FBD_BN(in_channels=in_channels, num_classes=num_classes)

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
            temp_model = get_resnet18_fbd_model(
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
                        norm: str = 'bn') -> Dict[str, Any]:
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
    # Store norm type for temp_model creation
    strategy.norm = norm
    
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
        model = get_resnet18_fbd_model(
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
                              norm: str = 'bn') -> Dict[str, Any]:
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
    # Store norm type for use in model creation
    strategy.norm = norm
    
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
        model = get_resnet18_fbd_model(
            norm=norm_type,
            in_channels=self.input_shape[0], 
            num_classes=self.num_classes
        ).to(self.device)
        
        model.load_from_dict(model_weights)
        return model
    
    def _compute_color_block_l2_distances(self, warehouse, ensemble_records, colors_ensemble):
        """
        Compute L2 distances between blocks with the same color across different positions.
        
        Args:
            warehouse: FBD warehouse containing function block weights
            ensemble_records: List of ensemble model records
            colors_ensemble: List of colors used in ensemble
            
        Returns:
            Dict: L2 distance statistics by position and color
        """
        model_parts = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
        
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
                print(f"[FBD Ensemble] Warning: Could not extract weights for color {color}: {e}")
                continue
        
        # Compute L2 distances between blocks of different colors at the same position
        l2_distances = {}
        position_comparisons = {}
        
        print(f"[FBD Ensemble] Computing L2 distances between different colors at same positions:")
        
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
                        
                        l2_distance = torch.norm(block1 - block2, p=2).item()
                        
                        pair_key = f"{color1}_vs_{color2}"
                        position_comparisons[position][pair_key] = l2_distance
                        position_distances.append(l2_distance)
            
            if position_distances:
                avg_distance = np.mean(position_distances)
                print(f"  {position}: Average L2 distance between colors = {avg_distance:.6f}")
                position_comparisons[position]['average'] = avg_distance
            
        # Also store the individual color blocks for potential future analysis
        for color in colors_ensemble:
            l2_distances[color] = {}
            for position in model_parts:
                if color_position_blocks[color][position]:
                    block = color_position_blocks[color][position][0]
                    l2_distances[color][position] = {
                        'norm': torch.norm(block, p=2).item(),  # L2 norm of the block itself
                        'shape': list(block.shape),
                        'num_params': block.numel()
                    }
        
        return {
            'by_color_and_position': l2_distances,
            'by_position_comparisons': position_comparisons,
            'colors_analyzed': colors_ensemble,
            'positions_analyzed': model_parts
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

def fbd_ensemble_evaluate(warehouse, 
                         round_num: int,
                         test_loader,
                         num_classes: int = 8,
                         input_shape: tuple = (1, 28, 28),
                         device: str = 'cpu',
                         norm: str = 'bn',
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
    # Store norm type for use in model creation
    strategy.norm = norm
    
    return strategy.evaluate(warehouse, round_num, num_ensemble, colors_ensemble) 