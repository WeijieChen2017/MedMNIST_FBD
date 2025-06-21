import os
import shutil
import hashlib
import medmnist
import torch
import torchvision.transforms as transforms
import PIL
from torch.utils.data import DataLoader, Subset, random_split
from medmnist import INFO

CACHE_DIR = "../medmnist-101/data_storage"
MEDMNIST_DIR = os.path.expanduser("/root/.medmnist")

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
                source_md5 = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                dest_md5 = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if source_md5 == dest_md5:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} and is the same")
                else:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} but is different, overwriting.")
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
                source_md5 = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                dest_md5 = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if source_md5 == dest_md5:
                    print(f"Dataset {dataset} already exists in cache and is the same")
                else:
                    print(f"Dataset {dataset} already exists in cache but is different, overwriting.")
                    shutil.copy(dest_npz_path, source_npz_path)

def load_data(data_flag, resize=False, as_rgb=False, download=True, size=28):
    """Load a MedMNIST dataset."""
    dataset_cache_name = f"{data_flag}" if size == 28 else f"{data_flag}_{size}"
    handle_dataset_cache(dataset_cache_name)

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Transformations
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

    # Load the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size)
    
    handle_dataset_cache(dataset_cache_name, post_execution=True)
    
    return train_dataset, val_dataset, test_dataset

def partition_data(dataset, num_clients, iid=False, data_flag=None):
    """Partitions a dataset into subsets for each client."""
    num_items = len(dataset)

    if not iid:
        # Default non-IID partitioning
        items_per_client = num_items // num_clients
        client_datasets = []
        all_indices = list(range(num_items))
        
        for i in range(num_clients):
            start_idx = i * items_per_client
            end_idx = (i + 1) * items_per_client if i != num_clients - 1 else num_items
            client_indices = all_indices[start_idx:end_idx]
            client_datasets.append(Subset(dataset, client_indices))
        return client_datasets
    else:
        # IID partitioning
        info = INFO[data_flag]
        task = info['task']
        
        if task == "multi-label, binary-class":
            # For multi-label, perform a random shuffle for IID
            all_indices = torch.randperm(num_items).tolist()
            items_per_client = num_items // num_clients
            client_datasets = []
            for i in range(num_clients):
                start_idx = i * items_per_client
                end_idx = (i + 1) * items_per_client if i != num_clients - 1 else num_items
                client_indices = all_indices[start_idx:end_idx]
                client_datasets.append(Subset(dataset, client_indices))
            return client_datasets
        else:
            # Stratified sampling for single-label classification
            labels = torch.tensor(dataset.labels)
            if labels.ndim > 1:
                labels = labels.squeeze()
            
            num_classes = len(torch.unique(labels))
            
            class_indices = [torch.where(labels == i)[0] for i in range(num_classes)]
            
            client_indices = [[] for _ in range(num_clients)]
            for indices_in_class in class_indices:
                indices_in_class = indices_in_class[torch.randperm(len(indices_in_class))]
                
                num_samples_per_client = len(indices_in_class) // num_clients
                for i in range(num_clients):
                    start = i * num_samples_per_client
                    end = (i + 1) * num_samples_per_client
                    client_indices[i].extend(indices_in_class[start:end].tolist())

                remaining = len(indices_in_class) % num_clients
                for i in range(remaining):
                    client_indices[i].append(indices_in_class[len(indices_in_class) - 1 - i].item())

            client_datasets = [Subset(dataset, indices) for indices in client_indices]
            return client_datasets

def get_data_loader(dataset, batch_size, shuffle=True):
    """Returns a DataLoader for a given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 