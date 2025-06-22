#!/usr/bin/env python3
"""
FBD Plan Generation Script

This script generates the shipping, request, and update plans for FBD training
based on the configuration settings in fbd_settings.py.

Usage:
    python generate_plans.py

Output:
    - shipping_plan.json
    - request_plan.json  
    - update_plan.json
"""

import json
from collections import defaultdict
from fbd_settings import (
    FBD_TRACE, FBD_INFO, MODEL_PARTS, 
    OUTER_ROUNDS_TOTAL, MODEL_PART_ORDER
)

def generate_model_to_blocks_mapping():
    """Generate mapping from model colors to their block IDs, ordered by layer."""
    # Create part ranking for sorting
    part_rank = {p: i for i, p in enumerate(MODEL_PART_ORDER)}
    
    # Group blocks by model color
    model_to_blocks = {mid: [] for mid in FBD_INFO["models"]}
    for bid, info in FBD_TRACE.items():
        model_to_blocks[info['color']].append(bid)
    
    # Sort blocks within each model by layer order
    for blist in model_to_blocks.values():
        blist.sort(key=lambda b: part_rank[FBD_TRACE[b]['model_part']])
    
    return model_to_blocks

def generate_plans():
    """Generate shipping, request, and update plans for all rounds."""
    print("Generating FBD plans...")
    
    # Get model-to-blocks mapping
    model_to_blocks = generate_model_to_blocks_mapping()
    
    # Initialize plan dictionaries
    shipping_plan = defaultdict(dict)
    request_plan = defaultdict(dict)
    update_plan = defaultdict(dict)
    
    print(f"Generating plans for {OUTER_ROUNDS_TOTAL} rounds...")
    
    # Generate plans for each round
    for outer_round in range(OUTER_ROUNDS_TOTAL):
        # Determine which micro-cycle we're in (0, 1, or 2)
        sched_idx = outer_round % 3
        schedule = FBD_INFO["training_plan"]["schedule"][sched_idx]
        
        round_num = outer_round + 1  # 1-based round numbering
        
        # Generate plans for each client in this round
        for client, active_model in schedule.items():
            # Get all models this client is involved in
            client_models = FBD_INFO["clients"][client]
            
            # Collect all blocks from all models this client is involved in
            all_client_blocks = []
            for model_id in client_models:
                all_client_blocks.extend(model_to_blocks[model_id])
            
            # Shipping and request plans: client needs all their blocks
            shipping_plan[round_num][client] = all_client_blocks
            request_plan[round_num][client] = all_client_blocks
            
            # Generate update plan
            update_plan[round_num][client] = generate_update_plan(
                client, active_model, client_models, model_to_blocks
            )
    
    return shipping_plan, request_plan, update_plan

def generate_update_plan(client, active_model, client_models, model_to_blocks):
    """Generate update plan for a specific client in a specific round."""
    # model_to_update: specify which blocks to update for each model part
    model_to_update = {}
    active_model_blocks = model_to_blocks[active_model]
    
    # Map each model part to its block ID for the active model
    for block_id in active_model_blocks:
        model_part = FBD_TRACE[block_id]['model_part']
        model_to_update[model_part] = block_id
    
    # model_as_regularizer: other models this client is involved in
    regularizer_models = []
    for model_id in client_models:
        if model_id != active_model:
            regularizer_model = {}
            regularizer_blocks = model_to_blocks[model_id]
            
            # Map each model part to its block ID for this regularizer model
            for block_id in regularizer_blocks:
                model_part = FBD_TRACE[block_id]['model_part']
                regularizer_model[model_part] = block_id
            
            regularizer_models.append(regularizer_model)
    
    return {
        "model_to_update": model_to_update,
        "model_as_regularizer": regularizer_models
    }

def save_plans_to_json(shipping_plan, request_plan, update_plan):
    """Save all plans to JSON files."""
    print("Saving plans to JSON files...")
    
    # Convert defaultdict keys to regular ints for JSON serialization
    plans = {
        "shipping_plan.json": {int(k): v for k, v in shipping_plan.items()},
        "request_plan.json": {int(k): v for k, v in request_plan.items()},
        "update_plan.json": {int(k): v for k, v in update_plan.items()}
    }
    
    for filename, plan_data in plans.items():
        with open(filename, "w") as f:
            json.dump(plan_data, f, indent=2)
        print(f"  ✓ Saved {filename}")

def print_plan_summary(shipping_plan, request_plan, update_plan):
    """Print a summary of the generated plans."""
    print("\n" + "="*50)
    print("PLAN GENERATION SUMMARY")
    print("="*50)
    
    print(f"Total rounds: {len(shipping_plan)}")
    print(f"Clients: {len(FBD_INFO['clients'])}")
    print(f"Models: {len(FBD_INFO['models'])}")
    print(f"Total blocks: {len(FBD_TRACE)}")
    
    # Show first round as example
    if 1 in update_plan:
        print(f"\nExample - Round 1 assignments:")
        for client, plan in update_plan[1].items():
            active_model_blocks = list(plan["model_to_update"].values())
            active_model_color = FBD_TRACE[active_model_blocks[0]]['color']
            reg_models = len(plan["model_as_regularizer"])
            print(f"  Client {client}: Train {active_model_color}, {reg_models} regularizers")

if __name__ == "__main__":
    try:
        # Import the required constants
        from fbd_settings import MODEL_PART_ORDER
        
        # Generate all plans
        shipping_plan, request_plan, update_plan = generate_plans()
        
        # Save to JSON files
        save_plans_to_json(shipping_plan, request_plan, update_plan)
        
        # Print summary
        print_plan_summary(shipping_plan, request_plan, update_plan)
        
        print(f"\n✅ Successfully generated FBD plans!")
        print("Files created:")
        print("  - shipping_plan.json")
        print("  - request_plan.json")
        print("  - update_plan.json")
        
    except Exception as e:
        print(f"❌ Error generating plans: {e}")
        raise 