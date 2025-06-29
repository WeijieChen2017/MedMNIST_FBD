"""
FBD Logic for Function Block Diversification
"""
import json
import copy
import importlib.util
import sys
import os
import torch
import torch.nn as nn
from collections import defaultdict
from argparse import Namespace

def load_fbd_settings(fbd_file_path):
    """
    Load FBD_TRACE, FBD_INFO, and TRANSPARENT_TO_CLIENT from a Python file.
    
    Args:
        fbd_file_path (str): Path to the FBD settings file (e.g., 'fbd_record/bloodmnist_plan_1.py')
        
    Returns:
        tuple: (FBD_TRACE, FBD_INFO, TRANSPARENT_TO_CLIENT) loaded from the file
    """
    # Convert relative path to absolute path
    abs_path = os.path.abspath(fbd_file_path)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("fbd_settings", abs_path)
    fbd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fbd_module)
    
    # Extract the settings
    fbd_trace = getattr(fbd_module, 'FBD_TRACE', {})
    fbd_info = getattr(fbd_module, 'FBD_INFO', {})
    transparent_to_client = getattr(fbd_module, 'TRANSPARENT_TO_CLIENT', True)  # Default to True for backward compatibility
    
    return fbd_trace, fbd_info, transparent_to_client

def load_shipping_plan(shipping_plan_path):
    """
    Load shipping plan from JSON file.
    
    Args:
        shipping_plan_path (str): Path to shipping plan JSON file
        
    Returns:
        dict: Shipping plan with round numbers as keys
    """
    with open(shipping_plan_path, 'r') as f:
        shipping_plan = json.load(f)
    
    # Convert string keys to integers for round numbers
    return {int(round_num): clients for round_num, clients in shipping_plan.items()}

def load_request_plan(request_plan_path):
    """
    Load request plan from JSON file.
    
    Args:
        request_plan_path (str): Path to request plan JSON file
        
    Returns:
        dict: Request plan with round numbers as keys
    """
    with open(request_plan_path, 'r') as f:
        request_plan = json.load(f)
    
    # Convert string keys to integers for round numbers
    return {int(round_num): clients for round_num, clients in request_plan.items()}

class FBDWarehouse:
    """
    Warehouse for storing and managing function block weights at the server.
    Organizes weights by FBD block IDs and enables flexible weight shipping/receiving.
    """
    
    def __init__(self, fbd_trace, model_template=None):
        """
        Initialize the warehouse.
        
        Args:
            fbd_trace (dict): FBD_TRACE dictionary mapping block IDs to model parts and colors
            model_template (nn.Module, optional): Template model to initialize weights from
        """
        self.fbd_trace = fbd_trace
        self.warehouse = {}  # Dictionary storing weights by block ID
        
        # Initialize warehouse with random weights or from template model
        self._initialize_warehouse(model_template)
    
    def _initialize_warehouse(self, model_template=None):
        """
        Initialize warehouse with weights for all function blocks.
        
        Args:
            model_template (nn.Module, optional): Template model to copy weights from
        """
        if model_template is not None:
            # Initialize from template model
            model_state = model_template.state_dict()
            
            for block_id, block_info in self.fbd_trace.items():
                model_part = block_info['model_part']
                
                # Extract weights for this model part
                part_weights = {}
                for param_name, param_tensor in model_state.items():
                    if param_name.startswith(model_part + '.'):
                        part_weights[param_name] = param_tensor.clone()
                
                self.warehouse[block_id] = part_weights
        else:
            # Initialize with empty dictionaries - will be populated later
            for block_id in self.fbd_trace.keys():
                self.warehouse[block_id] = {}
    
    def store_weights(self, block_id, state_dict):
        """
        Store weights for a specific function block.
        
        Args:
            block_id (str): FBD block ID (e.g., "AFA79")
            state_dict (dict): State dictionary containing the weights
        """
        if block_id not in self.fbd_trace:
            raise ValueError(f"Unknown block ID: {block_id}")
        
        # Debug: Check if weights are meaningful (not all zeros or identical)
        total_norm = 0.0
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # Convert integer tensors to float before computing norm
                if v.dtype in [torch.long, torch.int, torch.short, torch.uint8, torch.int32, torch.int64]:
                    # For integer tensors, convert to float temporarily for norm calculation
                    v_float = v.float()
                    total_norm += torch.norm(v_float).item()
                else:
                    # For floating point tensors, compute norm directly
                    total_norm += torch.norm(v).item()
        
        print(f"[WAREHOUSE DEBUG] Storing block {block_id}: total norm = {total_norm:.6f}")
        
        self.warehouse[block_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                   for k, v in state_dict.items()}
    
    def store_weights_batch(self, weights_dict):
        """
        Store weights for multiple function blocks at once.
        
        Args:
            weights_dict (dict): Dictionary mapping block IDs to their state_dicts
        """
        for block_id, state_dict in weights_dict.items():
            self.store_weights(block_id, state_dict)
    
    def retrieve_weights(self, block_id):
        """
        Retrieve weights for a specific function block.
        
        Args:
            block_id (str): FBD block ID
            
        Returns:
            dict: State dictionary containing the weights
        """
        if block_id not in self.warehouse:
            raise ValueError(f"Block ID not found in warehouse: {block_id}")
        
        return {k: v.clone() if isinstance(v, torch.Tensor) else v 
                for k, v in self.warehouse[block_id].items()}
    
    def retrieve_weights_batch(self, block_ids):
        """
        Retrieve weights for multiple function blocks.
        
        Args:
            block_ids (list): List of FBD block IDs
            
        Returns:
            dict: Dictionary mapping block IDs to their state_dicts
        """
        return {block_id: self.retrieve_weights(block_id) for block_id in block_ids}
    
    def get_model_weights(self, model_color):
        """
        Reconstruct complete model weights for a specific model (color).
        
        Args:
            model_color (str): Model color (e.g., "M0", "M1", etc.)
            
        Returns:
            dict: Complete state dictionary for the model organized by model parts
        """
        model_weights = {}
        
        # Find all blocks belonging to this model
        for block_id, block_info in self.fbd_trace.items():
            if block_info['color'] == model_color:
                model_part = block_info['model_part']
                block_weights = self.retrieve_weights(block_id)
                model_weights[model_part] = block_weights
        
        return model_weights
    
    def get_shipping_weights(self, shipping_list):
        """
        Prepare weights for shipping according to shipping plan.
        
        Args:
            shipping_list (list): List of block IDs to ship
            
        Returns:
            dict: Dictionary mapping model parts to their state_dicts
        """
        shipping_weights = {}
        
        for block_id in shipping_list:
            if block_id in self.fbd_trace:
                model_part = self.fbd_trace[block_id]['model_part']
                block_weights = self.retrieve_weights(block_id)
                shipping_weights[model_part] = block_weights
        
        return shipping_weights
    
    def warehouse_summary(self):
        """
        Get summary information about the warehouse contents.
        
        Returns:
            dict: Summary including block counts, model coverage, etc.
        """
        summary = {
            'total_blocks': len(self.warehouse),
            'models': defaultdict(list),
            'model_parts': defaultdict(list),
            'empty_blocks': []
        }
        
        for block_id, weights in self.warehouse.items():
            if block_id in self.fbd_trace:
                block_info = self.fbd_trace[block_id]
                color = block_info['color']
                model_part = block_info['model_part']
                
                summary['models'][color].append(block_id)
                summary['model_parts'][model_part].append(block_id)
                
                if not weights:
                    summary['empty_blocks'].append(block_id)
        
        return dict(summary)
    
    def save_warehouse(self, filepath):
        """
        Save warehouse state to file.
        
        Args:
            filepath (str): Path to save the warehouse
        """
        torch.save({
            'warehouse': self.warehouse,
            'fbd_trace': self.fbd_trace
        }, filepath)
    
    def load_warehouse(self, filepath):
        """
        Load warehouse state from file.
        
        Args:
            filepath (str): Path to load the warehouse from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.warehouse = checkpoint['warehouse']
        self.fbd_trace = checkpoint['fbd_trace']

def generate_client_model_palettes(num_clients, fbd_file_path):
    """
    Generate model palettes for each client based on FBD settings from file.
    
    Args:
        num_clients (int): Number of clients in the federated learning setup
        fbd_file_path (str): Path to the FBD settings file
        
    Returns:
        dict: Dictionary where keys are client IDs and values are their model palettes
    """
    # Load FBD settings from file
    fbd_trace, fbd_info, transparent_to_client = load_fbd_settings(fbd_file_path)
    
    client_model_palettes = {}
    
    for cid in range(num_clients):
        if cid in fbd_info["clients"]:
            # Get the models (colors) this client has access to
            client_colors = fbd_info["clients"][cid]
            
            # Create the model palette for this client
            model_palette = {}
            for fbd_id, fbd_entry in fbd_trace.items():
                if fbd_entry["color"] in client_colors:
                    # Make a deep copy to avoid modifying the original
                    palette_entry = copy.deepcopy(fbd_entry)
                    
                    # Remove color information if not transparent to client
                    if not transparent_to_client:
                        palette_entry.pop("color", None)
                    
                    model_palette[fbd_id] = palette_entry
            
            client_model_palettes[cid] = model_palette
        else:
            # Default: client has access to all models if not specified in plan
            default_palette = copy.deepcopy(fbd_trace)
            
            # Remove color information from all entries if not transparent to client
            if not transparent_to_client:
                for fbd_id, fbd_entry in default_palette.items():
                    fbd_entry.pop("color", None)
            
            client_model_palettes[cid] = default_palette
    
    return client_model_palettes

def load_config(data_flag: str, model_flag: str, size: int) -> Namespace:
    """Load configuration for a given dataset and model."""
    config_path = f"config/{data_flag}.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    for config in configs:
        if config['model_flag'] == model_flag and config['size'] == size:
            return Namespace(**config)
    
    raise ValueError(f"Configuration for model {model_flag} and size {size} not found in {config_path}") 