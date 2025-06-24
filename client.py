"""
Legacy Flower Client Implementation

Note: This file now contains only the legacy FlowerClient class for backward compatibility.
All FBD-related client functionality has been moved to fbd_client.py.

For active FBD federated learning, use:
- FBDFlowerClient (from fbd_client.py) - main Flower-based client
- FBDClient (from fbd_client.py) - standalone FBD client for testing
- train, test functions (from fbd_client.py) - training/testing logic
"""

from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import logging
from medmnist import INFO
import pickle
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Import FBD training functions
from fbd_client import train, test

# Disable Flower warnings
logging.getLogger("flwr").setLevel(logging.ERROR)

def log_gpu_memory(cid, device, phase):
    """Log the GPU memory usage for a specific client."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        print(f"[Client {cid}, {phase}] GPU Memory: {allocated:.2f} MB Allocated / {reserved:.2f} MB Reserved")

class FlowerClient(fl.client.Client):
    def __init__(self, cid, model, train_loader, val_loader, test_loader, data_flag, device):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_flag = data_flag
        self.device = device
        self.test_loader = test_loader
        # self.evaluator = Evaluator(data_flag, 'val')

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        parameters = ndarrays_to_parameters(ndarrays)
        return fl.common.GetParametersRes(status=fl.common.Status(code=fl.common.Code.OK, message="Success"), parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        # Deserialize parameters
        parameters_original = ins.parameters
        config = ins.config

        # Update local model with parameters received from the server
        if parameters_original.tensors:
            # Check if parameters are in the new dictionary format
            if parameters_original.tensor_type == "dict":
                state_dicts = pickle.loads(parameters_original.tensors[0])
                self.model.load_from_dict(state_dicts)
            # Fallback for initial parameters from the server
            else:
                ndarrays = parameters_to_ndarrays(parameters_original)
                params_dict = zip(self.model.state_dict().keys(), ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
        
        # Perform training
        lr = config["local_learning_rate"]
        current_update_plan = config.get("current_update_plan", None)
        round_num = config.get("server_round", None)
        train_result = train(self.model, self.train_loader, epochs=1, device=self.device, 
                    data_flag=self.data_flag, lr=lr, current_update_plan=current_update_plan, 
                    client_id=self.cid, round_num=round_num)
        
        # Handle different return types from train function
        if isinstance(train_result, tuple):
            loss, regularizer_metrics = train_result
        else:
            loss = train_result
            regularizer_metrics = None
        
        # Evaluate the new model
        train_loss, train_auc, train_acc = test(self.model, self.train_loader, device=self.device, data_flag=self.data_flag)
        val_loss, val_auc, val_acc = test(self.model, self.val_loader, device=self.device, data_flag=self.data_flag)
        test_loss, test_auc, test_acc = test(self.model, self.test_loader, device=self.device, data_flag=self.data_flag)

        metrics_dict = {
            "train_loss": loss, "val_loss": val_loss, "test_loss": test_loss,
            "train_auc": train_auc, "val_auc": val_auc, "test_auc": test_auc,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        }
        print(f"--> Client {self.cid} train_acc = {train_acc:.4f}, val_acc = {val_acc:.4f}, test_acc = {test_acc:.4f}")

        # Extract and serialize the requested layers
        requested_layers = config.get("requested_layers", [])
        if requested_layers:
            print(f"[Client {self.cid}] Sending layers: {requested_layers}")
            state_dicts_to_send = self.model.send_for_dict(requested_layers)
            serialized_weights = pickle.dumps(state_dicts_to_send)
            new_parameters = Parameters(tensors=[serialized_weights], tensor_type="dict")
        else:
            # Fallback to sending all parameters if none are specified
            ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            new_parameters = ndarrays_to_parameters(ndarrays)

        return FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=new_parameters,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics_dict
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        # Deserialize parameters
        parameters = ins.parameters
        if parameters.tensors:
            # Check if parameters are in the new dictionary format
            if parameters.tensor_type == "dict":
                state_dicts = pickle.loads(parameters.tensors[0])
                self.model.load_from_dict(state_dicts)
            # Fallback for initial parameters from the server
            else:
                ndarrays = parameters_to_ndarrays(parameters)
                params_dict = zip(self.model.state_dict().keys(), ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

        loss, auc, acc = test(self.model, self.val_loader, device=self.device, data_flag=self.data_flag)
        
        metrics = {"loss": float(loss), "auc": float(auc), "acc": float(acc)}
        
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.val_loader.dataset),
            metrics=metrics,
        )