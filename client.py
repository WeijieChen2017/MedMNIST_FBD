from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import logging
from medmnist import INFO
import pickle
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Disable Flower warnings
logging.getLogger("flwr").setLevel(logging.ERROR)
# from medmnist import Evaluator, INFO
from medmnist import INFO
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

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
        loss = train(self.model, self.train_loader, epochs=1, device=self.device, data_flag=self.data_flag, lr=lr)
        
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

def train(model, train_loader, epochs, device, data_flag, lr):
    """Train the model on the training set."""
    info = INFO[data_flag]
    task = info['task']
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    total_loss = 0
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def test(model, data_loader, device, data_flag):
    """Validate the model on the test set."""
    info = INFO[data_flag]
    task = info['task']
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    model.to(device)
    model.eval()
    total_loss = 0
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                targets_squeezed = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets_squeezed)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
                targets = targets.float().resize_(len(targets), 1)

            total_loss += loss.item()
            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets), 0)

    y_score = y_score.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    # calculate auc and acc
    if task == 'multi-label, binary-class':
        auc = roc_auc_score(y_true, y_score)
        # For multi-label, we need to threshold the predictions
        y_pred = (y_score > 0.5).astype(int)
        # Calculate accuracy for each label and average
        acc = np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    elif task == 'binary-class':
        auc = roc_auc_score(y_true, y_score[:, 1])  # Use probability of positive class
        acc = accuracy_score(y_true, y_score.argmax(axis=1))
    else:  # multi-class
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        acc = accuracy_score(y_true, y_score.argmax(axis=1))

    avg_loss = total_loss / len(data_loader)
    return [avg_loss, auc, acc]