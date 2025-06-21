import json
from argparse import Namespace

def load_config(data_flag: str, model_flag: str, size: int) -> Namespace:
    """Load configuration for a given dataset and model."""
    config_path = f"config/{data_flag}.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    for config in configs:
        if config['model_flag'] == model_flag and config['size'] == size:
            return Namespace(**config)
    
    raise ValueError(f"Configuration for model {model_flag} and size {size} not found in {config_path}") 