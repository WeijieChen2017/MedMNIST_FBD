"""
FBD Client implementation for Federated Learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import time
import os

from models import ResNet18_FBD_BN, ResNet18_FBD_IN, ResNet18_FBD_LN
from trainer import LocalTrainer
from communication import WeightTransfer
from fbd_logic import load_fbd_settings

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
                 norm: str = 'bn'):
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
        """
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Load FBD configuration
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        
        # Initialize local model
        self.model = get_resnet18_fbd_model(norm, input_shape[0], num_classes)
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
        
        # Use ResNet18_FBD's load_from_dict method
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
        
        # Extract weights using ResNet18_FBD's send_for_dict method
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