"""
FBD Federated Learning Main - Flower Integration
Integrates FBD (Function Block Diversification) with Flower federated learning framework
"""

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import argparse
import flwr as fl
import torch
import os
import json
import random
import numpy as np
import logging
from flwr.client import Client
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import Dict, List, Tuple, Optional
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar

# Import existing components
from fbd_dataset import load_data, partition_data, get_data_loader
from fbd_models import get_fbd_model
from medmnist import INFO
from fbd_utils import load_config

# Import FBD components
from fbd_utils import load_fbd_settings, load_shipping_plan, load_request_plan, FBDWarehouse, generate_client_model_palettes
from fbd_strategy import fbd_average_evaluate, fbd_comprehensive_evaluate, fbd_ensemble_evaluate, FBDStrategy
from fbd_communication import WeightTransfer

# Import pretrained weight loader
from fbd_root_ckpt import get_pretrained_fbd_model

# Import FBD client
from fbd_client import FBDFlowerClient


def get_fbd_model_with_pretrained(architecture: str, norm: str, in_channels: int, num_classes: int, use_imagenet: bool = False, device: str = 'cpu'):
    """Get the appropriate FBD model based on architecture and normalization type."""
    if use_imagenet:
        # Use ImageNet pretrained weights
        logging.info(f"🔄 Loading {architecture.upper()} FBD with ImageNet pretrained weights ({norm.upper()} normalization)")
        return get_pretrained_fbd_model(
            architecture=architecture,
            norm=norm,
            in_channels=in_channels,
            num_classes=num_classes,
            device=device,
            use_pretrained=True
        )
    else:
        # Use random initialization
        return get_fbd_model(architecture, norm, in_channels, num_classes)





def get_fit_config_fn(config):
    """Return a function which returns training configurations with learning rate scheduling."""
    def fit_config(server_round: int):
        # Learning rate schedule:
        # Rounds 1-20: original learning rate
        # Rounds 21-50: learning rate / 10
        # Rounds 51+: learning rate / 20
        
        base_lr = config.local_learning_rate
        
        if server_round <= 20:
            current_lr = base_lr
            schedule_info = "base rate"
        elif server_round <= 50:
            current_lr = base_lr / 10
            schedule_info = "1/10 rate"
        else:
            current_lr = base_lr / 20
            schedule_info = "1/20 rate"
        
        return {
            "local_learning_rate": current_lr,
            "server_round": server_round,
            "lr_schedule_info": schedule_info,
            "base_learning_rate": base_lr
        }
    return fit_config


def main():
    parser = argparse.ArgumentParser(description="FBD Federated Learning with Flower")
    parser.add_argument("--dataset", type=str, default="bloodmnist", help="MedMNIST dataset name")
    parser.add_argument("--model_flag", type=str, default="resnet18_fbd", help="Model to train")
    parser.add_argument("--size", type=int, default=28, help="Image size")
    parser.add_argument("--iid", action="store_true", help="Whether to partition data in an IID manner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--cpus_per_client", type=int, default=6, help="Number of CPUs allocated per client")
    parser.add_argument("--imagenet", action="store_true", help="Use ImageNet pretrained weights for initialization")
    
    # FBD specific arguments
    parser.add_argument("--fbd_config", type=str, default="fbd_record/fbd_settings.py", 
                       help="Path to FBD configuration file")
    parser.add_argument("--shipping_plan", type=str, default="shipping_plan.json",
                       help="Path to shipping plan JSON file")
    parser.add_argument("--request_plan", type=str, default="request_plan.json", 
                       help="Path to request plan JSON file")
    parser.add_argument("--update_plan", type=str, default="fbd_record/update_plan.json",
                       help="Path to update plan JSON file")
    parser.add_argument("--communication_dir", type=str, default="fbd_comm",
                       help="Directory for FBD communication files")
    parser.add_argument("--ensemble_size", type=int, default=None,
                       help="Number of ensemble models to generate (overrides config)")
    parser.add_argument("--ensemble_colors", type=str, nargs='+', default=None,
                       help="List of model colors for ensemble (e.g., M1 M2)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.dataset, args.model_flag.replace("_fbd", ""), args.size)
    
    # Override ensemble settings if provided via command line
    if args.ensemble_size is not None:
        config.num_ensemble = args.ensemble_size
    
    # Store original config rounds before any modifications
    original_config_rounds = config.num_rounds
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir is None:
        output_dir = os.path.join("runs", "fbd", args.dataset, str(args.size), str(args.seed))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Setup Logging ---
    log_file_path = os.path.join(output_dir, "simulation.log")
    
    # Configure the root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'),
                            logging.StreamHandler()
                        ])

    # Load model and data
    info = INFO[args.dataset]
    n_channels = 3 if config.as_rgb else info['n_channels']
    n_classes = len(info['label'])
    
    # Get normalization type from config (default to 'bn' if not specified)
    norm_type = getattr(config, 'norm', 'bn')
    
    # Extract architecture from model_flag (e.g., "resnet18_fbd" -> "resnet18")
    if '_fbd' in args.model_flag:
        architecture = args.model_flag.replace('_fbd', '')
    else:
        architecture = args.model_flag  # fallback if no _fbd suffix
    
    # Create model with optional ImageNet pretraining
    if args.imagenet:
        logging.info(f"🎯 ImageNet pretraining enabled for server model")
        model = get_fbd_model_with_pretrained(architecture, norm_type, n_channels, n_classes, use_imagenet=True, device=device)
    else:
        logging.info(f"🎯 Using random initialization for server model")
        model = get_fbd_model_with_pretrained(architecture, norm_type, n_channels, n_classes, use_imagenet=False, device=device)
        model = model.to(device)
    
    # Load REGULARIZER_PARAMS for configuration saving
    regularizer_params = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fbd_config", args.fbd_config)
        fbd_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fbd_config_module)
        regularizer_params = getattr(fbd_config_module, 'REGULARIZER_PARAMS', {})
    except Exception as e:
        logging.warning(f"Failed to load REGULARIZER_PARAMS for config: {e}")
    
    # Save configuration
    config_dict = {
        "dataset": args.dataset,
        "model_flag": args.model_flag,
        "normalization": norm_type,
        "actual_model_class": model.__class__.__name__,
        "imagenet_pretrained": args.imagenet,
        "initialization": "ImageNet pretrained" if args.imagenet else "Random initialization",
        "image_size": config.size,
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "num_rounds_note": "Derived from shipping plan, not config file",
        "original_config_rounds": original_config_rounds,
        "regularizer_params": regularizer_params,
        "batch_size": config.batch_size,
        "local_learning_rate": config.local_learning_rate,
        "num_ensemble": config.num_ensemble,
        "iid_partitioning": args.iid,
        "random_seed": args.seed,
        "output_directory": output_dir,
        "device": str(device),
        "fbd_config": args.fbd_config,
        "shipping_plan": args.shipping_plan,
        "request_plan": args.request_plan,
        "communication_dir": args.communication_dir
    }
    
    with open(os.path.join(output_dir, "fbd_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    
    logging.info(f"\n--- FBD Federated Learning Configuration ---")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.model_flag}")
    logging.info(f"Normalization: {norm_type}")
    logging.info(f"Actual Model Class: {model.__class__.__name__}")
    logging.info(f"Initialization: {'ImageNet pretrained' if args.imagenet else 'Random initialization'}")
    logging.info(f"Clients: {config.num_clients}")
    logging.info(f"Rounds: {config.num_rounds} (from shipping plan)")
    logging.info(f"Ensemble Models: {config.num_ensemble}")
    logging.info(f"Device: {device}")
    logging.info(f"FBD Config: {args.fbd_config}")
    logging.info(f"Output: {output_dir}")
    logging.info("--------------------------------------------\n")
    
    # Log the actual model being used
    logging.info(f"✅ Model instantiated: {model.__class__.__name__}")
    logging.info(f"   Normalization type: {norm_type}")
    logging.info(f"   Input channels: {n_channels}")
    logging.info(f"   Output classes: {n_classes}")
    logging.info(f"   Initialization: {'ImageNet pretrained' if args.imagenet else 'Random initialization'}")
    logging.info("")
    
    # Load and partition data
    train_dataset, val_dataset, test_dataset = load_data(
        args.dataset, config.resize, config.as_rgb, config.download, config.size
    )
    client_datasets = partition_data(train_dataset, config.num_clients, iid=args.iid, data_flag=args.dataset)
    
    # Load FBD settings and generate client palettes
    client_palettes = generate_client_model_palettes(config.num_clients, args.fbd_config)
    
    # Clean up communication directory
    if os.path.exists(args.communication_dir):
        import shutil
        shutil.rmtree(args.communication_dir)
    os.makedirs(args.communication_dir, exist_ok=True)
    
    # Load shipping, request, and update plans
    logging.info("Loading shipping, request, and update plans...")
    shipping_plan = load_shipping_plan(args.shipping_plan)
    request_plan = load_request_plan(args.request_plan)
    
    # Load update plan
    update_plan = None
    if args.update_plan and os.path.exists(args.update_plan):
        with open(args.update_plan, 'r') as f:
            update_plan = json.load(f)
    
    # FBD training is driven by shipping plan, not config
    max_shipping_round = max(shipping_plan.keys()) if shipping_plan else 0
    max_request_round = max(request_plan.keys()) if request_plan else 0
    fbd_total_rounds = max(max_shipping_round, max_request_round)
    
    if fbd_total_rounds <= 0:
        raise ValueError("No valid shipping/request plan found - cannot determine number of rounds")
    
    # Always use shipping plan rounds (ignore config.num_rounds completely)
    config.num_rounds = fbd_total_rounds
    
    logging.info(f"📋 FBD Training Configuration:")
    logging.info(f"   Shipping plan rounds: {len(shipping_plan)}")
    logging.info(f"   Request plan rounds: {len(request_plan)}")
    logging.info(f"   FBD training rounds: {config.num_rounds} (authoritative)")
    logging.info(f"   Config rounds (ignored): {original_config_rounds}")
    logging.info(f"   ✓ Training will stop after {config.num_rounds} rounds as defined by shipping plan")
    
    logging.info(f"Loaded shipping plan for {len(shipping_plan)} rounds")
    logging.info(f"Loaded request plan for {len(request_plan)} rounds")
    
    def client_fn(cid: str) -> Client:
        """Create an FBD Flower client."""
        client_dataset = client_datasets[int(cid)]
        train_loader = get_data_loader(client_dataset, config.batch_size)
        val_loader = get_data_loader(val_dataset, config.batch_size)
        test_loader = get_data_loader(test_dataset, config.batch_size)
        
        # Create client-specific model (clients use same initialization as server)
        if args.imagenet:
            client_model = get_fbd_model_with_pretrained(architecture, norm_type, n_channels, n_classes, use_imagenet=True, device=device)
        else:
            client_model = get_fbd_model_with_pretrained(architecture, norm_type, n_channels, n_classes, use_imagenet=False, device=device)
            client_model = client_model.to(device)
        
        # Get client palette
        client_palette = client_palettes.get(int(cid), {})
        
        return FBDFlowerClient(
            cid=int(cid),
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            data_flag=args.dataset,
            device=device,
            fbd_config_path=args.fbd_config,
            communication_dir=args.communication_dir,
            client_palette=client_palette,
            architecture=architecture
        ).to_client()
    
    # Initialize FBD strategy
    strategy = FBDStrategy(
        fbd_config_path=args.fbd_config,
        shipping_plan_path=args.shipping_plan,
        request_plan_path=args.request_plan,
        update_plan_path=args.update_plan,  # Pass update plan path
        num_clients=config.num_clients,
        communication_dir=args.communication_dir,
        model_template=model,
        output_dir=output_dir,
        num_classes=n_classes,
        input_shape=(n_channels, config.size, config.size),
        test_dataset=test_dataset,
        batch_size=config.batch_size,
        norm_type=norm_type,  # Pass normalization type
        architecture=architecture,  # Pass model architecture
        num_rounds=config.num_rounds,  # Pass total number of rounds
        num_ensemble=config.num_ensemble,  # Pass number of ensemble models from config
        ensemble_colors=args.ensemble_colors,  # Pass ensemble colors from command line
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Disable client-side evaluation
        min_fit_clients=config.num_clients,
        min_evaluate_clients=0,  # Disable client-side evaluation
        min_available_clients=config.num_clients,
        on_fit_config_fn=get_fit_config_fn(config)
    )
    
    # Define client resources
    gpus_per_client = 1 / (config.num_clients + 1) if device.type == "cuda" else 0
    client_resources = {"num_cpus": args.cpus_per_client, "num_gpus": gpus_per_client}
    
    logging.info(f"🚀 Starting FBD Federated Learning Simulation")
    logging.info(f"Clients: {config.num_clients}, Rounds: {config.num_rounds}")
    logging.info(f"GPU per client: {gpus_per_client:.3f}")
    logging.info(f"📄 Training progress will be saved to JSON files instead of verbose console output")
    
    # Temporarily reduce Flower's logging verbosity during simulation
    fl_logger = logging.getLogger("flwr")
    original_level = fl_logger.level
    fl_logger.setLevel(logging.WARNING)  # Suppress Flower's verbose output
    
    try:
        # Start Flower simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )
    finally:
        # Restore original logging level
        fl_logger.setLevel(original_level)
    
    # Save final results after training completion
    strategy.save_final_results()
    
    logging.info(f"\n🎉 FBD Federated Learning Completed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"📊 Best performance summary: {output_dir}/best_model_performance.csv")
    logging.info(f"💾 Best warehouse weights: {output_dir}/best_warehouse_weights.json")


if __name__ == "__main__":
    main() 