import flwr as fl
from client import test
from dataset import load_data, get_data_loader
import torch
from collections import OrderedDict
# from medmnist import Evaluator
import os
import time
from tensorboardX import SummaryWriter
from medmnist import INFO
from models import ResNet18, ResNet50
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import pandas as pd

OUTPUT_ROOT = "flwr_models"
EVALUATION_FILENAME = "evaluation_metrics.csv"
CENTRALIZED_EVALUATION_FILENAME = "centralized_evaluation_metrics.csv"
LOSS_PLOT_FILENAME = "loss_plot.png"

def get_evaluate_fn(model, data_flag, model_flag, device, num_rounds, output_dir, config):
    """Return an evaluation function for server-side evaluation."""
    info = INFO[data_flag]
    task = info['task']

    _, _, test_dataset = load_data(data_flag, config.resize, config.as_rgb, config.download, config.size)
    test_loader = get_data_loader(test_dataset, batch_size=128)
    # evaluator = Evaluator(data_flag, 'test')

    log_dir = os.path.join(output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, auc, acc = test(model, test_loader, device=device, data_flag=data_flag)
        print(f"Server evaluate at round {server_round}: data_flag = {data_flag}, loss = {loss:.4f}, auc = {auc:.4f}, acc = {acc:.4f}")

        # Always return both AUC and ACC metrics (loss is returned separately)
        metrics_dict = {
            "test_auc": auc,
            "test_acc": acc
        }

        writer.add_scalar(f'Loss/test_{data_flag}', loss, server_round)
        writer.add_scalar(f'AUC/test_{data_flag}', auc, server_round)
        writer.add_scalar(f'Accuracy/test_{data_flag}', acc, server_round)
        
        if server_round == num_rounds:
            model_output_dir = os.path.join(output_dir, "models")
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            
            save_path = os.path.join(model_output_dir, f'{model_flag}_final_model.pth')
            print(f"Saving final model to {save_path}")
            torch.save({'net': model.state_dict()}, save_path)
            writer.close()  

        return loss, metrics_dict

    return evaluate

def get_fit_config_fn(config):
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        fit_config = {
            "server_round": server_round,
            "local_epochs": 1,
            "local_learning_rate": config.local_learning_rate,
            "centralized_learning_rate": config.centralized_learning_rate,
        }
        return fit_config
    return fit_config 