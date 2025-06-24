"""
FBD Server implementation for Federated Learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import time
import os
import json

from fbd_models import get_fbd_model
from fbd_communication import WeightTransfer
from fbd_utils import (FBDWarehouse, load_fbd_settings, 
                       load_shipping_plan, load_request_plan)

class FBDServer:
    """
    FBD Federated Learning Server.
    Manages warehouse, coordinates clients, and orchestrates training rounds.
    """
    
    def __init__(self,
                 fbd_config_path: str,
                 shipping_plan_path: str,
                 request_plan_path: str,
                 num_clients: int = 6,
                 communication_dir: str = "fbd_comm",
                 num_classes: int = 8,
                 input_shape: tuple = (1, 28, 28),
                 device: str = 'cpu',
                 norm: str = 'bn',
                 architecture: str = 'resnet18'):
        """
        Initialize FBD server.
        
        Args:
            fbd_config_path: Path to FBD configuration file
            shipping_plan_path: Path to shipping plan JSON
            request_plan_path: Path to request plan JSON
            num_clients: Number of federated learning clients
            communication_dir: Directory for communication files
            num_classes: Number of output classes
            input_shape: Input tensor shape
            device: Device for computation
            norm: Normalization type ('bn', 'in', 'ln')
            architecture: Model architecture ('resnet18', 'resnet50')
        """
        self.device = device
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.norm = norm
        self.architecture = architecture
        
        # Load FBD configuration
        self.fbd_trace, self.fbd_info, self.transparent_to_client = load_fbd_settings(fbd_config_path)
        
        # Load shipping and request plans
        self.shipping_plan = load_shipping_plan(shipping_plan_path)
        self.request_plan = load_request_plan(request_plan_path)
        self.total_rounds = len(self.shipping_plan)
        
        # Initialize warehouse with template model using proper normalization
        template_model = get_fbd_model(
            architecture=architecture,
            norm=norm,
            in_channels=input_shape[0], 
            num_classes=num_classes
        )
        self.warehouse = FBDWarehouse(self.fbd_trace, template_model)
        
        # Initialize communication
        self.comm = WeightTransfer(communication_dir)
        
        # Server state
        self.current_round = 0
        self.round_results = []
        self.evaluation_results = []
        
        print(f"FBD Server initialized:")
        print(f"  Total rounds: {self.total_rounds}")
        print(f"  Clients: {num_clients}")
        print(f"  Function blocks: {len(self.fbd_trace)}")
        print(f"  Models: {list(self.fbd_info['models'].keys())}")
    
    def shipping_phase(self, round_num: int) -> Dict[int, List[str]]:
        """
        Execute shipping phase - send weights to all clients.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dict: Mapping client_id to list of shipped block IDs
        """
        print(f"\n=== Server Shipping Phase - Round {round_num} ===")
        
        if round_num not in self.shipping_plan:
            print(f"No shipping plan for round {round_num}")
            return {}
        
        round_shipping = self.shipping_plan[round_num]
        shipping_summary = {}
        
        for client_id in range(self.num_clients):
            client_str = str(client_id)
            if client_str in round_shipping:
                shipping_list = round_shipping[client_str]
                
                # Get weights from warehouse for shipping
                shipping_weights = self.warehouse.get_shipping_weights(shipping_list)
                
                # Send weights to client
                self.comm.server_send_weights(client_id, round_num, shipping_weights)
                
                shipping_summary[client_id] = shipping_list
                print(f"  Shipped {len(shipping_list)} blocks to client {client_id}")
            else:
                print(f"  No shipping plan for client {client_id}")
        
        return shipping_summary
    
    def collection_phase(self, round_num: int) -> Dict[int, List[str]]:
        """
        Execute collection phase - collect weights from all clients.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dict: Mapping client_id to list of collected block IDs
        """
        print(f"\n=== Server Collection Phase - Round {round_num} ===")
        
        if round_num not in self.request_plan:
            print(f"No request plan for round {round_num}")
            return {}
        
        round_requests = self.request_plan[round_num]
        collection_summary = {}
        
        # First, send request lists to all clients
        for client_id in range(self.num_clients):
            client_str = str(client_id)
            if client_str in round_requests:
                request_list = round_requests[client_str]
                self.comm.server_send_request_list(client_id, round_num, request_list)
                print(f"  Sent request for {len(request_list)} blocks to client {client_id}")
        
        # Then, collect responses from all clients
        for client_id in range(self.num_clients):
            client_str = str(client_id)
            if client_str in round_requests:
                try:
                    # Receive weights from client
                    received_weights, original_request = self.comm.server_receive_weights(client_id, round_num)
                    
                    # Store received weights in warehouse
                    self.warehouse.store_weights_batch(received_weights)
                    
                    collection_summary[client_id] = list(received_weights.keys())
                    print(f"  Collected {len(received_weights)} blocks from client {client_id}")
                    
                except Exception as e:
                    print(f"  Failed to collect from client {client_id}: {e}")
        
        return collection_summary
    
    def evaluation_phase(self, round_num: int) -> Dict[str, Any]:
        """
        Execute evaluation phase - evaluate models (placeholder implementation).
        
        Args:
            round_num: Current round number
            
        Returns:
            Dict: Evaluation results
        """
        print(f"\n=== Server Evaluation Phase - Round {round_num} ===")
        
        # Placeholder evaluation - reconstruct models and get basic statistics
        evaluation_results = {
            'round': round_num,
            'timestamp': time.time(),
            'models': {}
        }
        
        for model_color in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
            try:
                # Reconstruct model from warehouse
                model_weights = self.warehouse.get_model_weights(model_color)
                
                # Create temporary model for evaluation with proper normalization
                norm_type = getattr(self, 'norm', 'bn')
                architecture_type = getattr(self, 'architecture', 'resnet18')
                temp_model = get_fbd_model(
                    architecture=architecture_type,
                    norm=norm_type,
                    in_channels=self.input_shape[0], 
                    num_classes=self.num_classes
                )
                temp_model.load_from_dict(model_weights)
                
                # Basic model statistics (placeholder)
                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
                
                model_eval = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_parts': list(model_weights.keys()),
                    'part_count': len(model_weights),
                    # TODO: Add actual evaluation metrics (accuracy, loss, etc.)
                    'placeholder_accuracy': 85.0 + round_num * 0.5,  # Simulated improvement
                    'placeholder_loss': 0.5 - round_num * 0.01  # Simulated loss reduction
                }
                
                evaluation_results['models'][model_color] = model_eval
                print(f"  Model {model_color}: {model_eval['part_count']} parts, "
                      f"{model_eval['placeholder_accuracy']:.1f}% accuracy")
                
            except Exception as e:
                print(f"  Failed to evaluate model {model_color}: {e}")
                evaluation_results['models'][model_color] = {'error': str(e)}
        
        self.evaluation_results.append(evaluation_results)
        return evaluation_results
    
    def run_round(self, round_num: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute one complete round of federated learning.
        
        Args:
            round_num: Round number to execute
            verbose: Whether to print detailed progress
            
        Returns:
            Dict: Complete round results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"SERVER ROUND {round_num}/{self.total_rounds}")
            print(f"{'='*60}")
        
        round_start_time = time.time()
        
        # Phase 1: Shipping
        shipping_summary = self.shipping_phase(round_num)
        
        # Wait for clients to complete training (simulation)
        if verbose:
            print(f"\n  Waiting for clients to complete training...")
        time.sleep(1)  # Simulate training time
        
        # Phase 2: Collection
        collection_summary = self.collection_phase(round_num)
        
        # Phase 3: Evaluation
        evaluation_results = self.evaluation_phase(round_num)
        
        round_time = time.time() - round_start_time
        
        round_results = {
            'round': round_num,
            'round_time': round_time,
            'shipping_summary': shipping_summary,
            'collection_summary': collection_summary,
            'evaluation_results': evaluation_results,
            'warehouse_stats': self.warehouse.warehouse_summary()
        }
        
        self.round_results.append(round_results)
        self.current_round = round_num
        
        if verbose:
            print(f"\nRound {round_num} completed in {round_time:.2f}s")
            print(f"  Shipped to {len(shipping_summary)} clients")
            print(f"  Collected from {len(collection_summary)} clients")
            print(f"  Evaluated {len(evaluation_results['models'])} models")
        
        return round_results
    
    def run_federated_learning(self, 
                              start_round: int = 1,
                              end_round: Optional[int] = None,
                              verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run complete federated learning process.
        
        Args:
            start_round: Starting round number
            end_round: Ending round number (if None, run all rounds)
            verbose: Whether to print progress
            
        Returns:
            List: Results for all executed rounds
        """
        if end_round is None:
            end_round = self.total_rounds
        
        if verbose:
            print(f"\n🚀 Starting FBD Federated Learning")
            print(f"   Rounds: {start_round} to {end_round}")
            print(f"   Clients: {self.num_clients}")
            print(f"   Function Blocks: {len(self.fbd_trace)}")
        
        start_time = time.time()
        executed_rounds = []
        
        for round_num in range(start_round, end_round + 1):
            try:
                round_results = self.run_round(round_num, verbose=verbose)
                executed_rounds.append(round_results)
                
                # Clean up communication files for this round
                self.comm.cleanup_round(round_num)
                
            except Exception as e:
                print(f"❌ Error in round {round_num}: {e}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n🎉 Federated Learning Completed!")
            print(f"   Executed rounds: {len(executed_rounds)}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per round: {total_time/len(executed_rounds):.2f}s")
        
        return executed_rounds
    
    def get_server_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive server summary.
        
        Returns:
            Dict: Server state and statistics
        """
        return {
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'num_clients': self.num_clients,
            'completed_rounds': len(self.round_results),
            'warehouse_summary': self.warehouse.warehouse_summary(),
            'communication_stats': self.comm.get_communication_stats(),
            'latest_evaluation': self.evaluation_results[-1] if self.evaluation_results else None
        }
    
    def save_server_state(self, filepath: str):
        """
        Save complete server state.
        
        Args:
            filepath: Path to save server state
        """
        # Save warehouse separately
        warehouse_path = filepath.replace('.pth', '_warehouse.pth')
        self.warehouse.save_warehouse(warehouse_path)
        
        # Save server state
        state = {
            'current_round': self.current_round,
            'round_results': self.round_results,
            'evaluation_results': self.evaluation_results,
            'server_config': {
                'num_clients': self.num_clients,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'total_rounds': self.total_rounds
            }
        }
        
        torch.save(state, filepath)
        print(f"Server state saved to {filepath}")
        print(f"Warehouse saved to {warehouse_path}")
    
    def load_server_state(self, filepath: str):
        """
        Load server state.
        
        Args:
            filepath: Path to load server state from
        """
        # Load server state
        state = torch.load(filepath, map_location=self.device)
        
        self.current_round = state['current_round']
        self.round_results = state['round_results']
        self.evaluation_results = state['evaluation_results']
        
        # Load warehouse
        warehouse_path = filepath.replace('.pth', '_warehouse.pth')
        self.warehouse.load_warehouse(warehouse_path)
        
        print(f"Server state loaded from {filepath}")
    
    def save_results_summary(self, output_dir: str):
        """
        Save detailed results summary to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save round results
        with open(os.path.join(output_dir, 'round_results.json'), 'w') as f:
            json.dump(self.round_results, f, indent=2, default=str)
        
        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save server summary
        with open(os.path.join(output_dir, 'server_summary.json'), 'w') as f:
            json.dump(self.get_server_summary(), f, indent=2, default=str)
        
        print(f"Results saved to {output_dir}") 