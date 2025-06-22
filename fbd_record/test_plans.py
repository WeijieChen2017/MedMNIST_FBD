#!/usr/bin/env python3
"""
Test script for FBD plans

This script tests the generated FBD plans and shows usage examples.

Usage:
    python test_plans.py
"""

import json
import os
from fbd_settings import FBD_TRACE, FBD_INFO

def load_plans():
    """Load the generated plan files."""
    plans = {}
    plan_files = ['shipping_plan.json', 'request_plan.json', 'update_plan.json']
    
    for filename in plan_files:
        if not os.path.exists(filename):
            print(f"❌ Missing {filename}. Run generate_plans.py first.")
            return None
        
        with open(filename, 'r') as f:
            plan_name = filename.replace('.json', '').replace('_', ' ').title()
            plans[filename] = json.load(f)
            print(f"✓ Loaded {plan_name}: {len(plans[filename])} rounds")
    
    return plans

def test_plan_consistency(plans):
    """Test that plans are consistent and valid."""
    print("\n" + "="*50)
    print("TESTING PLAN CONSISTENCY")
    print("="*50)
    
    shipping_plan = plans['shipping_plan.json']
    request_plan = plans['request_plan.json']
    update_plan = plans['update_plan.json']
    
    errors = []
    
    # Test 1: All plans have same rounds
    rounds_shipping = set(shipping_plan.keys())
    rounds_request = set(request_plan.keys())
    rounds_update = set(update_plan.keys())
    
    if not (rounds_shipping == rounds_request == rounds_update):
        errors.append("Plans have different round numbers")
    else:
        print(f"✓ All plans have {len(rounds_shipping)} rounds")
    
    # Test 2: All clients appear in each round
    expected_clients = set(str(c) for c in FBD_INFO['clients'].keys())
    
    for round_num in rounds_shipping:
        for plan_name, plan_data in [('shipping', shipping_plan), ('request', request_plan), ('update', update_plan)]:
            round_clients = set(plan_data[round_num].keys())
            if round_clients != expected_clients:
                errors.append(f"Round {round_num} {plan_name} plan missing clients: {expected_clients - round_clients}")
    
    if not errors:
        print("✓ All clients appear in all rounds for all plans")
    
    # Test 3: Update plans have valid block IDs
    all_block_ids = set(FBD_TRACE.keys())
    
    for round_num, round_data in update_plan.items():
        for client, client_plan in round_data.items():
            # Check model_to_update blocks
            update_blocks = set(client_plan['model_to_update'].values())
            invalid_blocks = update_blocks - all_block_ids
            if invalid_blocks:
                errors.append(f"Round {round_num} Client {client} has invalid update blocks: {invalid_blocks}")
            
            # Check regularizer blocks
            for reg_model in client_plan['model_as_regularizer']:
                reg_blocks = set(reg_model.values())
                invalid_reg_blocks = reg_blocks - all_block_ids
                if invalid_reg_blocks:
                    errors.append(f"Round {round_num} Client {client} has invalid regularizer blocks: {invalid_reg_blocks}")
    
    if not errors:
        print("✓ All block IDs in update plans are valid")
    
    # Print any errors found
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All consistency tests passed!")
        return True

def show_example_usage(plans):
    """Show examples of how to use the plans."""
    print("\n" + "="*50)
    print("EXAMPLE USAGE")
    print("="*50)
    
    update_plan = plans['update_plan.json']
    
    # Show examples for first few rounds
    for round_num in ['1', '2', '3']:
        if round_num in update_plan:
            print(f"\n--- Round {round_num} ---")
            
            for client, plan in update_plan[round_num].items():
                # Get the model being trained
                model_blocks = list(plan['model_to_update'].values())
                if model_blocks:
                    active_model = FBD_TRACE[model_blocks[0]]['color']
                    
                    # Get regularizer models
                    reg_models = []
                    for reg_plan in plan['model_as_regularizer']:
                        reg_blocks = list(reg_plan.values())
                        if reg_blocks:
                            reg_model = FBD_TRACE[reg_blocks[0]]['color']
                            reg_models.append(reg_model)
                    
                    print(f"  Client {client}: Train {active_model}, Regularize with {reg_models}")

def show_statistics(plans):
    """Show statistics about the plans."""
    print("\n" + "="*50)
    print("PLAN STATISTICS")
    print("="*50)
    
    update_plan = plans['update_plan.json']
    
    # Count how many times each model is trained
    model_training_count = {}
    total_rounds = len(update_plan)
    
    for round_data in update_plan.values():
        for client_plan in round_data.values():
            model_blocks = list(client_plan['model_to_update'].values())
            if model_blocks:
                active_model = FBD_TRACE[model_blocks[0]]['color']
                model_training_count[active_model] = model_training_count.get(active_model, 0) + 1
    
    print(f"Model training frequency over {total_rounds} rounds:")
    for model, count in sorted(model_training_count.items()):
        percentage = (count / total_rounds) * 100
        print(f"  {model}: {count} times ({percentage:.1f}%)")
    
    # Check balance
    counts = list(model_training_count.values())
    if len(set(counts)) == 1:
        print("✓ All models are trained equally")
    else:
        print(f"❌ Unbalanced training: min={min(counts)}, max={max(counts)}")

if __name__ == "__main__":
    print("Testing FBD Plans")
    print("="*50)
    
    # Load plans
    plans = load_plans()
    if plans is None:
        exit(1)
    
    # Run tests
    if test_plan_consistency(plans):
        show_example_usage(plans)
        show_statistics(plans)
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")
        exit(1) 