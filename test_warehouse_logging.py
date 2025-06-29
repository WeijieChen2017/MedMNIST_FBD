#!/usr/bin/env python3
"""
Test script to demonstrate FBD Warehouse logging functionality.
This script creates a warehouse, performs some operations, and shows the log output.
"""

import torch
import torch.nn as nn
import os
import sys

# Add the current directory to the path to import FBD modules
sys.path.append('.')

from fbd_utils import FBDWarehouse
from fbd_models import get_fbd_model

def test_warehouse_logging():
    """Test warehouse logging functionality."""
    print("🧪 Testing FBD Warehouse Logging...")
    
    # Create a simple FBD trace for testing
    test_fbd_trace = {
        'AFA79': {'model_part': 'in_layer', 'color': 'M0'},
        'BFB81': {'model_part': 'layer1', 'color': 'M1'},
        'CFC83': {'model_part': 'layer2', 'color': 'M2'},
        'DFD85': {'model_part': 'layer3', 'color': 'M3'},
        'EFE87': {'model_part': 'layer4', 'color': 'M4'},
        'FFF89': {'model_part': 'out_layer', 'color': 'M5'}
    }
    
    # Create a template model
    template_model = get_fbd_model(
        architecture='resnet18',
        norm='bn',
        in_channels=1,
        num_classes=8
    )
    
    # Create warehouse with logging (log will be saved to test_warehouse.log)
    log_file_path = "test_warehouse.log"
    print(f"📝 Creating warehouse with log file: {log_file_path}")
    
    warehouse = FBDWarehouse(test_fbd_trace, template_model, log_file_path)
    
    # Perform some warehouse operations to generate logs
    print("\n📦 Performing warehouse operations...")
    
    # Test storing some weights
    test_weights = {
        'in_layer.conv1.weight': torch.randn(64, 1, 7, 7),
        'in_layer.conv1.bias': torch.randn(64),
        'in_layer.bn1.weight': torch.ones(64),
        'in_layer.bn1.bias': torch.zeros(64),
        'in_layer.bn1.running_mean': torch.zeros(64).long(),  # Integer tensor to test dtype handling
        'in_layer.bn1.running_var': torch.ones(64).long()     # Integer tensor to test dtype handling
    }
    
    warehouse.store_weights('AFA79', test_weights)
    
    # Test retrieving weights
    retrieved_weights = warehouse.retrieve_weights('AFA79')
    print(f"✅ Retrieved {len(retrieved_weights)} parameters from block AFA79")
    
    # Test getting model weights
    model_weights = warehouse.get_model_weights('M0')
    print(f"✅ Reconstructed model M0 with {len(model_weights)} parts")
    
    # Test batch operations
    batch_weights = {
        'BFB81': test_weights,
        'CFC83': test_weights
    }
    warehouse.store_weights_batch(batch_weights)
    print(f"✅ Stored batch of {len(batch_weights)} blocks")
    
    print(f"\n📋 Warehouse logging complete! Check the log file: {log_file_path}")
    
    # Show a summary of the log file
    if os.path.exists(log_file_path):
        print(f"\n📄 Log file contents ({log_file_path}):")
        print("-" * 60)
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                print(f"{line_num:2}: {line.rstrip()}")
        print("-" * 60)
    else:
        print(f"❌ Log file not found: {log_file_path}")

if __name__ == "__main__":
    test_warehouse_logging() 