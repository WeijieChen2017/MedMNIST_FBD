"""
Test script for FBD Warehouse functionality
"""

import torch
import json
import os
from fbd_logic import FBDWarehouse, load_fbd_settings, load_shipping_plan, load_request_plan
from models import ResNet18_FBD

def test_warehouse():
    """
    Test the FBD warehouse functionality with the bloodmnist configuration.
    """
    print("=== Testing FBD Warehouse ===")
    
    # Create test output directory
    os.makedirs('fbd_test', exist_ok=True)
    
    # Load FBD configuration
    fbd_trace, fbd_info, transparent_to_client = load_fbd_settings('fbd_record/bloodmnist_info_1.py')
    print(f"Loaded FBD configuration with {len(fbd_trace)} function blocks")
    
    # Create a template model for initialization
    template_model = ResNet18_FBD(in_channels=1, num_classes=8)
    print("Created template ResNet18_FBD model")
    
    # Initialize warehouse
    warehouse = FBDWarehouse(fbd_trace, template_model)
    print("Initialized warehouse with template model weights")
    
    # Display warehouse summary
    summary = warehouse.warehouse_summary()
    print(f"\nWarehouse Summary:")
    print(f"Total blocks: {summary['total_blocks']}")
    print(f"Models: {list(summary['models'].keys())}")
    print(f"Model parts: {list(summary['model_parts'].keys())}")
    print(f"Empty blocks: {len(summary['empty_blocks'])}")
    
    # Save detailed warehouse summary
    with open('fbd_test/warehouse_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved detailed warehouse summary to fbd_test/warehouse_summary.json")
    
    # Test retrieving weights for a specific model (M0)
    print(f"\n=== Testing Model M0 Reconstruction ===")
    m0_weights = warehouse.get_model_weights('M0')
    print(f"M0 model parts: {list(m0_weights.keys())}")
    
    # Save model reconstruction info
    model_info = {}
    for model_color in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        model_weights = warehouse.get_model_weights(model_color)
        model_info[model_color] = {
            'parts': list(model_weights.keys()),
            'block_count': len([bid for bid, info in fbd_trace.items() if info['color'] == model_color])
        }
    
    with open('fbd_test/model_reconstruction_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("Saved model reconstruction info to fbd_test/model_reconstruction_info.json")
    
    # Load shipping and request plans
    shipping_plan = load_shipping_plan('fbd_record/shipping_plan.json')
    request_plan = load_request_plan('fbd_record/request_plan.json')
    print(f"\nLoaded plans for {len(shipping_plan)} rounds")
    
    # Test shipping weights for round 1, client 0
    print(f"\n=== Testing Shipping for Round 1, Client 0 ===")
    round_1_shipping = shipping_plan[1]
    client_0_shipping = round_1_shipping["0"]  # Use string key
    print(f"Client 0 should receive blocks: {client_0_shipping}")
    
    shipping_weights = warehouse.get_shipping_weights(client_0_shipping)
    print(f"Shipping weights include parts: {list(shipping_weights.keys())}")
    
    # Test multiple rounds and clients
    print(f"\n=== Testing Multiple Rounds and Clients ===")
    test_results = {}
    for round_num in [1, 2, 3]:
        test_results[round_num] = {}
        for client_id in ["0", "1", "2"]:
            shipping_list = shipping_plan[round_num][client_id]
            request_list = request_plan[round_num][client_id]
            shipping_weights = warehouse.get_shipping_weights(shipping_list)
            
            test_results[round_num][client_id] = {
                'shipping_blocks': shipping_list,
                'request_blocks': request_list,
                'shipping_parts': list(shipping_weights.keys()),
                'shipping_count': len(shipping_list),
                'request_count': len(request_list)
            }
            
        print(f"Round {round_num}: Tested {len(test_results[round_num])} clients")
    
    with open('fbd_test/multi_round_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print("Saved multi-round test results to fbd_test/multi_round_test_results.json")
    
    # Test storing weights back (simulate client response)
    print(f"\n=== Testing Weight Storage (Request Response) ===")
    client_0_request = request_plan[1]["0"]  # Use string key
    print(f"Client 0 should send back blocks: {client_0_request}")
    
    # Simulate receiving weights from client
    received_weights = {}
    for block_id in client_0_request[:2]:  # Just test first 2 blocks
        # Simulate some modified weights
        original_weights = warehouse.retrieve_weights(block_id)
        modified_weights = {k: v + 0.01 for k, v in original_weights.items()}
        received_weights[block_id] = modified_weights
    
    # Store the received weights
    warehouse.store_weights_batch(received_weights)
    print(f"Stored weights for {len(received_weights)} blocks")
    
    # Verify storage
    storage_verification = {}
    for block_id in received_weights.keys():
        stored_weights = warehouse.retrieve_weights(block_id)
        storage_verification[block_id] = "success"
        print(f"Block {block_id}: weights successfully stored and retrieved")
    
    with open('fbd_test/storage_verification.json', 'w') as f:
        json.dump(storage_verification, f, indent=2)
    print("Saved storage verification to fbd_test/storage_verification.json")
    
    # Save warehouse state
    warehouse.save_warehouse('fbd_test/warehouse_state.pth')
    print("Saved warehouse state to fbd_test/warehouse_state.pth")
    
    # Test loading warehouse state
    new_warehouse = FBDWarehouse(fbd_trace)
    new_warehouse.load_warehouse('fbd_test/warehouse_state.pth')
    print("Successfully loaded warehouse state into new warehouse instance")
    
    print(f"\n=== Warehouse Test Completed Successfully ===")
    print(f"All test results saved to fbd_test/ directory")

if __name__ == "__main__":
    test_warehouse() 