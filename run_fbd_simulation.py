#!/usr/bin/env python3
"""
Simple script to run FBD federated learning simulation with Flower
"""

import subprocess
import sys
import os
import time
import shutil
import glob
import logging
from fbd_utils import load_config
from fbd_models import get_model_info

current_time = time.strftime("%m%d_%H%M", time.localtime(time.time() - 5*60*60))

OUTPUT_DIR = f"col5_fbd/bloodmnist_{current_time}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# At the top, add configuration
ARCHITECTURE = "resnet18"  # ← CHANGE THIS for ResNet18 or ResNet50
NORMALIZATION = "bn"       # ← CHANGE THIS for different normalization
FLWR_COMM_DIR = f"fbd_flower_comm_{current_time}"

if not os.path.exists(FLWR_COMM_DIR):
    os.makedirs(FLWR_COMM_DIR)

# Load FBD configuration settings

def load_fbd_settings():
    """Load FBD settings from the FBD configuration file."""
    try:
        # Import the FBD configuration module
        sys.path.append('fbd_record')
        import fbd_record.fbd_settings as fbd_settings_module
        return {
            'epochs_per_stage': fbd_settings_module.EPOCHS_PER_STAGE,
            'blocks_per_stage': fbd_settings_module.BLOCKS_PER_STAGE,
            'ensemble_size': fbd_settings_module.ENSEMBLE_SIZE,
            'ensemble_colors': fbd_settings_module.ENSEMBLE_COLORS,
            'regularizer_params': fbd_settings_module.REGULARIZER_PARAMS
        }
    except ImportError as e:
        raise ImportError(f"Could not load FBD configuration from fbd_record/fbd_settings.py: {e}")
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

def cleanup_ray_sessions():
    """
    Clean up old Ray session directories to prevent disk space issues.
    This removes all Ray session directories in /tmp/ray/ that are older than current session.
    """
    try:
        ray_tmp_dir = "/tmp/ray"
        if not os.path.exists(ray_tmp_dir):
            print("No Ray temp directory found - skipping cleanup")
            return
        
        # Get all session directories
        session_dirs = glob.glob(os.path.join(ray_tmp_dir, "session_*"))
        
        if not session_dirs:
            print("No Ray session directories found - skipping cleanup")
            return
        
        print("🧹 Cleaning up Ray session directories...")
        
        # Get current time for comparison
        current_time = time.time()
        cleaned_count = 0
        total_size_cleaned = 0
        
        for session_dir in session_dirs:
            try:
                # Check if directory is older than 1 hour (3600 seconds) or if it's not being used
                dir_mtime = os.path.getmtime(session_dir)
                if current_time - dir_mtime > 3600:  # Older than 1 hour
                    # Calculate directory size before deletion
                    dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(session_dir)
                                 for filename in filenames)
                    
                    # Remove the directory
                    shutil.rmtree(session_dir)
                    cleaned_count += 1
                    total_size_cleaned += dir_size
                    print(f"   Removed: {os.path.basename(session_dir)} ({dir_size / (1024**3):.2f} GB)")
                    
            except (OSError, PermissionError) as e:
                # Some directories might be in use - skip them
                print(f"   Skipped {session_dir}: {e}")
                continue
        
        if cleaned_count > 0:
            print(f"✅ Cleaned {cleaned_count} Ray session directories, freed {total_size_cleaned / (1024**3):.2f} GB")
        else:
            print("✅ No old Ray session directories to clean")
            
    except Exception as e:
        print(f"Warning: Failed to cleanup Ray sessions: {e}")


def get_model_name_from_norm(architecture, norm_type):
    """Get the correct model name based on normalization type."""
    try:
        info = get_model_info(architecture, norm_type)
        return info['description']
    except:
        return f"{architecture.upper()}_FBD_{norm_type.upper()}"

def run_fbd_simulation():
    """Run FBD federated learning simulation with predefined parameters."""
    
    # Clean up old Ray session directories before starting
    cleanup_ray_sessions()
    print()  # Add spacing after cleanup
    
    # Check if we're in the right directory
    if not os.path.exists("fbd_main.py"):
        print("❌ Error: fbd_main.py not found. Please run from the correct directory.")
        return False
    
    # Check required files
    required_files = [
        "fbd_record/fbd_settings.py",
        "fbd_record/shipping_plan.json", 
        "fbd_record/request_plan.json"
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
        "--cpus_per_client", "2",
        "--fbd_config", "fbd_record/fbd_settings.py",
        "--shipping_plan", "fbd_record/shipping_plan.json",
        "--request_plan", "fbd_record/request_plan.json",
        "--update_plan", "fbd_record/update_plan.json",
        "--communication_dir", FLWR_COMM_DIR,
        "--imagenet",
        "--output_dir", OUTPUT_DIR,
        "--ensemble_size", str(fbd_settings['ensemble_size']),
        "--ensemble_colors"] + fbd_settings['ensemble_colors']
    
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
    print(f"  Regularizer type: {fbd_settings['regularizer_params'].get('type', 'N/A')}")
    print(f"  Regularizer distance: {fbd_settings['regularizer_params'].get('distance_type', 'N/A')}")
    print(f"  Regularizer coefficient: {fbd_settings['regularizer_params'].get('coefficient', 'N/A')}")
    print(f"  Seed: 42")
    print(f"  CPUs per client: {config.cpus_per_client}")
    print(f"  Communication dir: {FLWR_COMM_DIR}")
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