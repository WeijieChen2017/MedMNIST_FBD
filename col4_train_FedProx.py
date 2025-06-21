import os
import argparse
import subprocess
import shutil
import hashlib

DATASETS_2D = [
    "bloodmnist", # no specific rules
    "breastmnist", # --as_rgb
    "chestmnist", # --as_rgb
    "dermamnist", # no specific rules
    "octmnist", # --as_rgb
    "organamnist", # --as_rgb
    "organcmnist", # --as_rgb
    "organsmnist", # --as_rgb
    "pathmnist", # no specific rules
    "pneumoniamnist", # --as_rgb
    "retinamnist", # no specific rules
    "tissuemnist", # --as_rgb
]

fold_0 = [
    # "bloodmnist",
    # "breastmnist",
    "chestmnist",
]

fold_1 = [
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
]

fold_2 = [
    "organamnist",
    "organcmnist",
    "organsmnist",
]

fold_3 = [
    "pathmnist",
    "retinamnist",
    "tissuemnist",
]

# Define different mu values to try for FedProx
MU_VALUES = [0.01, 0.1, 1.0]  # Different proximal term weights to experiment with

MODEL_INDEX = [1]
RANDOM_SEEDS = {1: 42, 2: 426, 3: 729}
CACHE_DIR = "data_storage"
MEDMNIST_DIR = os.path.expanduser("/root/.medmnist")
BATCH_SIZE_2D_DEFAULT = 128 
BATCH_SIZE_EXPAND_FACTOR_2D = 12

def handle_dataset_cache(dataset, post_execution=False):
    """Manages the dataset cache by copying files when needed."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(MEDMNIST_DIR):
        os.makedirs(MEDMNIST_DIR)
        
    source_npz_path = os.path.join(CACHE_DIR, f"{dataset}.npz")
    dest_npz_path = os.path.join(MEDMNIST_DIR, f"{dataset}.npz")

    if not post_execution:
        print(f"Looking for {dataset} in {CACHE_DIR}")
        # Before execution: if file is in cache but not in destination, copy it.
        if os.path.exists(source_npz_path):
            if not os.path.exists(dest_npz_path):
                print(f"Found {dataset} in cache. Copying to {MEDMNIST_DIR}")
                shutil.copy(source_npz_path, dest_npz_path)
            else:
                # please have the md5 check here
                md5_hash = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if md5_hash == hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest():
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} and is the same")
                else:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} but is different")
                    shutil.copy(source_npz_path, dest_npz_path)
        else:
            print(f"Dataset {dataset} not found in cache {CACHE_DIR}")
    else:
        # Copy downloaded dataset to cache if not already there
        if os.path.exists(dest_npz_path):
            if not os.path.exists(source_npz_path):
                print(f"Copying {dataset} from {MEDMNIST_DIR} to cache")
                shutil.copy(dest_npz_path, source_npz_path)
            else:
                md5_hash = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                if md5_hash == hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest():
                    print(f"Dataset {dataset} already exists in cache and is the same")
                else:
                    print(f"Dataset {dataset} already exists in cache but is different")
                    shutil.copy(dest_npz_path, source_npz_path)

def main():
    """Run MedMNIST 2D training for all datasets."""
    parser = argparse.ArgumentParser(description="Run MedMNIST 2D training for all datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Run for a specific dataset (optional).")
    parser.add_argument("--fold", type=int, default=-1, help="Fold number (default: 1).")
    parser.add_argument("--mu", type=float, default=None, help="Specific mu value to use (optional).")
    args = parser.parse_args()

    # Determine which datasets to run
    if args.dataset:
        if args.dataset in DATASETS_2D:
            datasets_to_run = [args.dataset]
            print(f"Running 2D training for {args.dataset}")
        else:
            print(f"Error: Dataset '{args.dataset}' not found in 2D datasets.")
            return
    else:
        if args.fold == 0:
            datasets_to_run = fold_0
        elif args.fold == 1:
            datasets_to_run = fold_1
        elif args.fold == 2:
            datasets_to_run = fold_2
        elif args.fold == 3:
            datasets_to_run = fold_3
        else:
            datasets_to_run = DATASETS_2D
        print(f"Running 2D training for all datasets: {datasets_to_run}")

    # Create fedprox_2d directory if it doesn't exist
    fedprox_dir = "col4_train_FedProx"
    if not os.path.exists(fedprox_dir):
        os.makedirs(fedprox_dir)

    print(f"================================================")
    # 2D commands - only image size 28
    IMAGE_SIZE = 28
    MODEL_NAMES = ["resnet18", "resnet50"]
    script_path = "main.py"
    
    # Determine which mu values to use
    mu_values_to_try = [args.mu] if args.mu is not None else MU_VALUES
    
    for model_index in MODEL_INDEX:
        for dataset in datasets_to_run:
            for model_name in MODEL_NAMES:
                for mu in mu_values_to_try:
                    # Create output directory for this specific training
                    output_dir = os.path.join(fedprox_dir, f"{dataset}/{model_name}/{IMAGE_SIZE}/{model_index}/mu_{mu}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    command = (
                        f"python {script_path} "
                        f"--iid "
                        f"--dataset {dataset} "
                        f"--model_flag {model_name} --size {IMAGE_SIZE} "
                        f"--seed {RANDOM_SEEDS[model_index]} "
                        f"--output_dir {output_dir} "
                        f"--strategy fedprox "
                        f"--mu {mu}"
                    )
                    
                    # print(f"Previewing command: {command}")

                    handle_dataset_cache(dataset)
                    print(f"Executing: {command}")
                    subprocess.run(command, shell=True, check=True)
                    handle_dataset_cache(dataset, post_execution=True)
                    print(f"Results saved to: {output_dir}")
                    print(f"--------------------------------")
    
    print(f"================================================")
    print(f"All 2D training completed. Results saved in {fedprox_dir}/ directory.")

if __name__ == "__main__":
    main()