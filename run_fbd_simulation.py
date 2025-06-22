#!/usr/bin/env python3
"""
Simple script to run FBD federated learning simulation with Flower
"""

import subprocess
import sys
import os
import time
from config_loader import load_config
from models import get_model_info

# At the top, add configuration
ARCHITECTURE = "resnet50"  # ← CHANGE THIS for ResNet50
NORMALIZATION = "bn"       # ← CHANGE THIS for different normalization

# Load FBD configuration settings

def load_fbd_settings():
    """Load FBD settings from the FBD configuration file."""
    try:
        # Import the FBD configuration module
        sys.path.append('fbd_record')
        import fbd_record.bloodmnist_info_1 as bloodmnist_info_1
        return {
            'epochs_per_stage': bloodmnist_info_1.EPOCHS_PER_STAGE,
            'blocks_per_stage': bloodmnist_info_1.blocks_per_stage,
            'ensemble_size': bloodmnist_info_1.ENSEMBLE_SIZE,
            'ensemble_colors': bloodmnist_info_1.ENSEMBLE_COLORS
        }
    except ImportError as e:
        raise ImportError(f"Could not load FBD configuration from fbd_record/bloodmnist_info_1.py: {e}")
    except AttributeError as e:
        raise AttributeError(f"Missing required attribute in FBD configuration: {e}")

# Load FBD settings
fbd_settings = load_fbd_settings()
blocks_per_stage = fbd_settings['blocks_per_stage']
blocks_per_stage_str = "".join(map(str, blocks_per_stage))

# I want like 0618_1424 as the current time
# the time shown is 0618_2149, but current time is 0618_1649
# this is because the time zone is different
# so the time delay is 5 hours

current_time = time.strftime("%m%d_%H%M", time.localtime(time.time() - 5*60*60))

OUTPUT_DIR = f"col5_fbd/bloodmnist_{current_time}_{blocks_per_stage_str}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_model_name_from_norm(architecture, norm_type):
    """Get the correct model name based on normalization type."""
    try:
        info = get_model_info(architecture, norm_type)
        return info['description']
    except:
        return f"{architecture.upper()}_FBD_{norm_type.upper()}"

def run_fbd_simulation():
    """Run FBD federated learning simulation with predefined parameters."""
    
    # Check if we're in the right directory
    if not os.path.exists("fbd_main.py"):
        print("❌ Error: fbd_main.py not found. Please run from the correct directory.")
        return False
    
    # Check required files
    required_files = [
        "fbd_record/bloodmnist_info_1.py",
        "shipping_plan.json", 
        "request_plan.json"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Error: Required file {file} not found.")
            return False
    
    print("🚀 Starting FBD Federated Learning Simulation with Flower")
    print("📊 Evaluation includes: Comprehensive (M0-M5 + Averaging) + Ensemble (configurable size)")
    print("=" * 60)
    
    # Define simulation parameters
    cmd = [
        "python3", "fbd_main.py",
        "--dataset", "bloodmnist",
        "--model_flag", f"{ARCHITECTURE}_fbd",
        "--size", "28",
        "--seed", "42",
        "--cpus_per_client", "6",
        "--fbd_config", "fbd_record/bloodmnist_info_1.py",
        "--shipping_plan", "shipping_plan.json",
        "--request_plan", "request_plan.json",
        "--communication_dir", "fbd_flower_comm",
        "--imagenet",
        "--output_dir", OUTPUT_DIR,
    ]
    
    print("Command:", " ".join(cmd))

    # Load configuration to display details
    try:
        config = load_config("bloodmnist", ARCHITECTURE, 28)
        # Override num_ensemble with value from FBD configuration
        config.num_ensemble = fbd_settings['ensemble_size']
    except ValueError as e:
        print(f"❌ Error loading configuration: {e}")
        return False

    print("\n" + "="*30)
    print("       Training Plan")
    print("="*30)
    print(f"  Dataset: bloodmnist")
    norm_type = getattr(config, 'norm', 'bn')
    print(f"  Model: {get_model_name_from_norm(ARCHITECTURE, norm_type)}")
    print(f"  Normalization: {norm_type}")
    print(f"  Image size: {config.size}x{config.size}")
    print(f"  Number of clients: {config.num_clients}")
    print(f"  Number of rounds: {config.num_rounds}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Local learning rate: {config.local_learning_rate}")
    print(f"  Epochs per stage: {fbd_settings['epochs_per_stage']}")
    print(f"  Blocks per stage: {fbd_settings['blocks_per_stage']}")
    print(f"  Ensemble models: {config.num_ensemble}")
    print(f"  Ensemble colors: {fbd_settings['ensemble_colors']}")
    print(f"  Seed: 42")
    print(f"  CPUs per client: 6")
    print(f"  Communication dir: fbd_flower_comm")
    print(f"  Using ImageNet pretrained weights: True")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("="*30)
    print()

    # User confirmation
    confirm = input("👉 Proceed with training? (Y/n): ").lower().strip()
    if confirm == 'n':
        print("🛑 Simulation cancelled by user.")
        return False
    
    try:
        # Run the simulation
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n🎉 FBD Simulation completed successfully!")
        print("Check the output directory: ", OUTPUT_DIR)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Simulation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    print("FBD Federated Learning Simulation Runner")
    print("========================================")
    
    success = run_fbd_simulation()
    
    if success:
        print("\n✅ All done! Check the results in the output directory.")
        sys.exit(0)
    else:
        print("\n❌ Simulation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 