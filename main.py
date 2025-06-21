import argparse
import flwr as fl
import torch
from flwr.client import Client
import os
import json
import time
import random
import numpy as np

from client import FlowerClient
from dataset import load_data, partition_data, get_data_loader
from models import ResNet18, ResNet50
from server import get_evaluate_fn
from medmnist import INFO
from config_loader import load_config
from custom_strategy import CsvLoggingFedAvg, CsvLoggingFedProx

def get_fit_config_fn(config, strategy_name, mu=None):
    """Return a function which returns training configurations."""
    def fit_config(server_round: int):
        """Return training configuration dict for each round.
        
        Perform if/else to different rounds' config.
        """
        fit_config_dict = {
            "local_learning_rate": config.local_learning_rate,
            "server_round": server_round,
            "layers_to_send": ["layer1", "layer4", "out_layer"]
        }
        if strategy_name == "fedprox":
            fit_config_dict["mu"] = mu
        return fit_config_dict
    return fit_config

def main():
    parser = argparse.ArgumentParser(description="Flower MedMNIST")
    parser.add_argument("--dataset", type=str, required=True, help="MedMNIST dataset name")
    parser.add_argument("--model_flag", type=str, default="resnet50", choices=["resnet18", "resnet50"], help="Model to train")
    parser.add_argument("--size", type=int, default=28, help="Image size")
    parser.add_argument("--iid", action="store_true", help="Whether to partition data in an IID manner.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--cpus_per_client", type=int, default=6, help="Number of CPUs allocated per client (default: 4)")
    parser.add_argument("--mu", type=float, default=0.1, help="FedProx proximal term weight (default: 0.1)")
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="Federated learning strategy to use")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.dataset, args.model_flag, args.size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create output directory for results
    if args.output_dir is None:
        # Create default output directory following the naming pattern
        output_dir = os.path.join("runs", args.dataset, args.model_flag, str(args.size), str(args.seed))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration to config.json
    config_dict = {
        "dataset": args.dataset,
        "model_flag": args.model_flag,
        "image_size": config.size,
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "batch_size": config.batch_size,
        "download": config.download,
        "resize": config.resize,
        "as_rgb": config.as_rgb,
        "local_learning_rate": config.local_learning_rate,
        "centralized_learning_rate": config.centralized_learning_rate,
        "iid_partitioning": args.iid,
        "random_seed": args.seed,
        "output_directory": output_dir,
        "device": str(device),
        "cpus_per_client": args.cpus_per_client,
        "strategy": args.strategy,
        "mu": args.mu if args.strategy == "fedprox" else None
    }
    
    config_file_path = os.path.join(output_dir, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Configuration saved to {config_file_path}")

    # Print configuration details
    print("\n--- Configuration Details ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_flag}")
    print(f"Image size: {config.size}")
    print(f"Number of clients: {config.num_clients}")
    print(f"Number of rounds: {config.num_rounds}")
    print(f"Batch size: {config.batch_size}")
    print(f"Download: {config.download}")
    print(f"Resize: {config.resize}")
    print(f"As RGB: {config.as_rgb}")
    print(f"Local learning rate: {config.local_learning_rate}")
    print(f"Centralized learning rate: {config.centralized_learning_rate}")
    print(f"IID partitioning: {args.iid}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"CPUs per client: {args.cpus_per_client}")
    print(f"Strategy: {args.strategy}")
    if args.strategy == "fedprox":
        print(f"FedProx mu: {args.mu}")
    print("-----------------------------\n")

    # Load model
    info = INFO[args.dataset]
    n_channels = 3 if config.as_rgb else info['n_channels']
    n_classes = len(info['label'])
    if args.model_flag == "resnet18":
        model = ResNet18_FBD(in_channels=n_channels, num_classes=n_classes)
    elif args.model_flag == "resnet50":
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {args.model_flag}")
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_data(args.dataset, config.resize, config.as_rgb, config.download, config.size)
    client_datasets = partition_data(train_dataset, config.num_clients, iid=args.iid, data_flag=args.dataset)

    # Print dataset statistics
    print("\n--- Dataset Statistics ---")
    print(f"Dataset: {args.dataset}")
    print(f"Total training samples: {len(train_dataset)}")
    
    task = info['task']
    print(f"Number of classes: {n_classes}")
    print(f"Task type: {task}")
    
    labels = train_dataset.labels
    if task == 'multi-label, binary-class':
        # For multi-label, count occurrences for each label
        labels_tensor = torch.tensor(labels)
        print("Label distribution:")
        for i in range(n_classes):
            num_samples = torch.sum(labels_tensor[:, i]).item()
            label_name = info['label'][str(i)]
            print(f"  Label '{label_name}': {int(num_samples)} samples")
    else:
        # For single-label, count occurrences of each class
        labels_tensor = torch.tensor(labels).squeeze()
        unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, counts):
            label_name = info['label'][str(label.item())]
            print(f"  Label '{label_name}' ({label.item()}): {count.item()} samples")
    
    print("--------------------------\n")

    # Print client dataset statistics
    print("\n--- Client Dataset Statistics ---")
    client_distributions = {}
    for cid in range(config.num_clients):
        client_dataset = client_datasets[cid]
        client_labels = client_dataset.dataset.labels[client_dataset.indices]
        
        client_distributions[f"client_{cid}"] = {"num_samples": len(client_labels)}
        
        print(f"\nClient {cid}:")
        print(f"  Total samples: {len(client_labels)}")
        
        label_dist = {}
        if task == 'multi-label, binary-class':
            labels_tensor = torch.tensor(client_labels)
            print("  Label distribution:")
            for i in range(n_classes):
                num_samples = torch.sum(labels_tensor[:, i]).item()
                label_name = info['label'][str(i)]
                label_dist[label_name] = int(num_samples)
                print(f"    Label '{label_name}': {int(num_samples)} samples")
        else:
            labels_tensor = torch.tensor(client_labels).squeeze()
            unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
            print("  Label distribution:")
            for label, count in zip(unique_labels, counts):
                label_name = info['label'][str(label.item())]
                label_dist[f"{label_name} ({label.item()})"] = count.item()
                print(f"    Label '{label_name}' ({label.item()}): {count.item()} samples")
        client_distributions[f"client_{cid}"]["label_distribution"] = label_dist
            
    # Save client distributions to a file
    split_file_path = os.path.join(output_dir, f"{args.dataset}_data_split.json")
    with open(split_file_path, "w") as f:
        json.dump(client_distributions, f, indent=4)
    print(f"\nClient data distribution saved to {split_file_path}")
    print("---------------------------------\n")

    # --- FBD Global Namespace and Mappings ---
    num_clients = config.num_clients
    fbd_id_counter = 0
    client_mappings = {}
    server_structure_map = {block_name: [] for block_name in FBD_BLOCK_NAMES}

    for cid in range(num_clients):
        client_map = {}
        for block_name in FBD_BLOCK_NAMES:
            fbd_id = f"fbd_{fbd_id_counter}"
            client_map[fbd_id] = block_name
            server_structure_map[block_name].append(fbd_id)
            fbd_id_counter += 1
        client_mappings[str(cid)] = client_map
    
    # Pass the server_structure_map to the strategy
    # The initial parameters are sent before the strategy is fully initialized,
    # so we pickle it here to be included in the initial parameters.
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )
    initial_parameters.tensor_type = "initial"
    
    def client_fn(cid: str) -> Client:
        """Create a Flower client."""
        client_dataset = client_datasets[int(cid)]
        train_loader = get_data_loader(client_dataset, config.batch_size)
        val_loader = get_data_loader(val_dataset, config.batch_size)
        test_loader = get_data_loader(test_dataset, config.batch_size)
        
        # Each client gets its own model and its unique FBD mapping
        client_model = model
        fbd_mapping = client_mappings[cid]

        return FlowerClient(cid, client_model, train_loader, val_loader, test_loader, args.dataset, device, fbd_mapping).to_client()

    # Choose strategy based on argument
    if args.strategy == "fedavg":
        strategy = CsvLoggingFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            evaluate_fn=get_evaluate_fn(model, args.dataset, args.model_flag, device, config.num_rounds, output_dir, config),
            on_fit_config_fn=get_fit_config_fn(config, args.strategy),
            output_dir=output_dir,
        )
    else:  # fedprox
        strategy = CsvLoggingFedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            evaluate_fn=get_evaluate_fn(model, args.dataset, args.model_flag, device, config.num_rounds, output_dir, config),
            on_fit_config_fn=get_fit_config_fn(config, args.strategy, args.mu),
            output_dir=output_dir,
            mu=args.mu
        )

    # Define client resources
    gpus_per_client = 1 / (config.num_clients + 1) if device.type == "cuda" else 0
    client_resources = {"num_cpus": args.cpus_per_client, "num_gpus": gpus_per_client}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    main()
