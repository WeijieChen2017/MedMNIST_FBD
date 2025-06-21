"""
Local training implementation for FBD Federated Learning clients
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import time

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
                   verbose: bool = False) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            data_loader: Data loader (if None, uses mock data)
            batch_size: Batch size for mock data
            num_batches: Number of batches for mock data
            verbose: Whether to print training progress
            
        Returns:
            Dict: Training metrics for this epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Use mock data if no data loader provided
        if data_loader is None:
            data_loader = self.generate_mock_data(batch_size, num_batches)
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats) 