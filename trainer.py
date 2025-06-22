"""
Local training implementation for FBD Federated Learning clients
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import time

# Import regularizer configuration
from fbd_record.fbd_settings import REGULARIZER_PARAMS

class LocalTrainer:
    """
    Handles local training for FBD clients with mock data.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize local trainer.
        
        Args:
            model: The neural network model to train
            num_classes: Number of output classes
            input_shape: Input tensor shape (channels, height, width)
            learning_rate: Learning rate for optimizer
            device: Device to run training on
        """
        self.model = model
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'total_samples': 0,
            'loss_history': [],
            'accuracy_history': []
        }
    
    def generate_mock_data(self, batch_size: int = 32, num_batches: int = 10):
        """
        Generate mock training data for testing.
        
        Args:
            batch_size: Number of samples per batch
            num_batches: Number of batches to generate
            
        Yields:
            tuple: (inputs, targets) for each batch
        """
        for _ in range(num_batches):
            # Generate random input data
            inputs = torch.randn(batch_size, *self.input_shape).to(self.device)
            
            # Generate random targets
            targets = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
            
            yield inputs, targets
    
    def train_epoch(self, 
                   data_loader=None, 
                   batch_size: int = 32, 
                   num_batches: int = 10,
                   verbose: bool = False,
                   current_update_plan=None,
                   client_id=None,
                   round_num=None) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            data_loader: Data loader (if None, generates mock data)
            batch_size: Batch size for mock data
            num_batches: Number of batches to train
            verbose: Whether to print training progress
            current_update_plan: Client-specific update plan for current round
            client_id: Client identifier
            round_num: Current training round
            
        Returns:
            Dict: Training metrics (loss, accuracy, etc.)
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Check if we have current update plan for regularized training
        use_update_plan = (current_update_plan is not None and 
                          client_id is not None and 
                          round_num is not None)
        
        # Use mock data if no data loader provided
        if data_loader is None:
            data_loader = self.generate_mock_data(batch_size, num_batches)
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            if use_update_plan:
                # Get current update plan for this client
                model_to_update_parts = current_update_plan["model_to_update"]
                model_as_regularizer_list = current_update_plan["model_as_regularizer"]
                
                # 1. Build the model_to_update from the main model
                model_to_update = self._extract_model_parts(self.model, model_to_update_parts)
                
                # 2. Build the optimizer for the model_to_update
                model_to_update_optimizer = torch.optim.Adam(model_to_update.parameters(), lr=self.optimizer.param_groups[0]['lr'])
                
                # 3. Load regularizer models from shipped weights
                regularizer_models = self._load_regularizer_models(model_as_regularizer_list, self.model, inputs.device)
                
                # Forward pass through model_to_update
                outputs_main = model_to_update(inputs)
                
                # Compute base loss
                base_loss = self.criterion(outputs_main, targets)
                
                # Determine regularizer type and compute regularized loss
                # Use configuration from REGULARIZER_PARAMS
                regularizer_type = REGULARIZER_PARAMS["type"]
                regularization_strength = REGULARIZER_PARAMS["coefficient"]
                
                if regularizer_type == "weights":
                    # Compute weight distance regularization
                    weight_regularizer = self._compute_weight_regularizer(model_to_update, regularizer_models)
                    loss = base_loss + regularization_strength * weight_regularizer
                    
                elif regularizer_type == "consistency loss":
                    # Compute output consistency regularization
                    consistency_regularizer = self._compute_consistency_regularizer(
                        model_to_update, regularizer_models, inputs, inputs.device
                    )
                    loss = base_loss + regularization_strength * consistency_regularizer
                
                else:
                    loss = base_loss
                
                if verbose and batch_idx == 0:  # Only print once per epoch
                    print(f"[Client {client_id}] Round {round_num}: Using {regularizer_type} regularization with {len(model_as_regularizer_list)} regularizers, loss={loss.item():.4f}")
                
                # Use the specialized optimizer for model_to_update
                model_to_update_optimizer.zero_grad()
                loss.backward()
                model_to_update_optimizer.step()
                
                # Update the main model with the trained model_to_update parts
                self._update_main_model_from_parts(self.model, model_to_update, model_to_update_parts)
                
            else:
                # Standard training without update plan
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Standard training step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            if use_update_plan:
                _, predicted = torch.max(outputs_main.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            
            if verbose and (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Calculate metrics
        avg_loss = epoch_loss / num_batches
        accuracy = 100.0 * epoch_correct / epoch_total
        
        # Update statistics
        self.training_stats['total_epochs'] += 1
        self.training_stats['total_batches'] += num_batches
        self.training_stats['total_samples'] += epoch_total
        self.training_stats['loss_history'].append(avg_loss)
        self.training_stats['accuracy_history'].append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': epoch_total,
            'batches': num_batches
        }
    
    def train_multiple_epochs(self, 
                            epochs: int = 1,
                            batch_size: int = 32,
                            num_batches: int = 10,
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Train model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for mock data
            num_batches: Number of batches per epoch
            verbose: Whether to print training progress
            
        Returns:
            Dict: Training summary
        """
        start_time = time.time()
        epoch_results = []
        
        if verbose:
            print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}:")
            
            epoch_metrics = self.train_epoch(
                batch_size=batch_size,
                num_batches=num_batches,
                verbose=verbose
            )
            
            epoch_results.append(epoch_metrics)
            
            if verbose:
                print(f"  Loss: {epoch_metrics['loss']:.4f}, "
                      f"Accuracy: {epoch_metrics['accuracy']:.2f}%")
        
        training_time = time.time() - start_time
        
        # Calculate summary statistics
        avg_loss = np.mean([r['loss'] for r in epoch_results])
        final_accuracy = epoch_results[-1]['accuracy']
        total_samples = sum(r['samples'] for r in epoch_results)
        
        summary = {
            'epochs': epochs,
            'training_time': training_time,
            'avg_loss': avg_loss,
            'final_accuracy': final_accuracy,
            'total_samples': total_samples,
            'epoch_results': epoch_results,
            'final_stats': self.training_stats.copy()
        }
        
        if verbose:
            print(f"Training completed in {training_time:.2f}s")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Final accuracy: {final_accuracy:.2f}%")
        
        return summary
    
    def evaluate_model(self, 
                      data_loader=None,
                      batch_size: int = 32,
                      num_batches: int = 5) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data_loader: Data loader (if None, uses mock data)
            batch_size: Batch size for mock data
            num_batches: Number of batches for evaluation
            
        Returns:
            Dict: Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            if data_loader is None:
                data_loader = self.generate_mock_data(batch_size, num_batches)
            
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get detailed training statistics.
        
        Returns:
            Dict: Training statistics
        """
        return self.training_stats.copy()
    
    def reset_stats(self):
        """Reset training statistics."""
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'total_samples': 0,
            'loss_history': [],
            'accuracy_history': []
        }
    
    def save_model(self, filepath: str):
        """
        Save model state.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load model state.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

    def _extract_model_parts(self, model, model_to_update_parts):
        """
        Extract specific model parts to create model_to_update.
        
        Args:
            model: The main model
            model_to_update_parts: Dict mapping layer names to FBD block IDs
            
        Returns:
            nn.Module: A model containing only the specified parts
        """
        # Create a new model with the same architecture
        import copy
        model_to_update = copy.deepcopy(model)
        
        # Note: For a more sophisticated implementation, you could create a custom
        # model that only contains the specified layers, but for now we use the full model
        # and rely on the optimizer to only update the relevant parts
        
        return model_to_update

    def _load_regularizer_models(self, model_as_regularizer_list, template_model, device):
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

    def _compute_weight_regularizer(self, model_to_update, regularizer_models):
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

    def _compute_consistency_regularizer(self, model_to_update, regularizer_models, inputs, device):
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

    def _update_main_model_from_parts(self, main_model, model_to_update, model_to_update_parts):
        """
        Update the main model with trained parts from model_to_update.
        
        Args:
            main_model: The main model to update
            model_to_update: The trained model parts
            model_to_update_parts: Dict mapping layer names to FBD block IDs
        """
        # For now, we update the entire main model with the trained model_to_update
        # In a more sophisticated implementation, you would only update the specific parts
        # specified in model_to_update_parts
        
        main_model.load_state_dict(model_to_update.state_dict()) 