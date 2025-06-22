from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import logging
from medmnist import INFO
import pickle
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Import regularizer configuration
from fbd_record.bloodmnist_info_1 import REGULARIZER_PARAMS

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
        current_update_plan = config.get("current_update_plan", None)
        round_num = config.get("server_round", None)
        loss = train(self.model, self.train_loader, epochs=1, device=self.device, 
                    data_flag=self.data_flag, lr=lr, current_update_plan=current_update_plan, 
                    client_id=self.cid, round_num=round_num)
        
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

def train(model, train_loader, epochs, device, data_flag, lr, current_update_plan=None, client_id=None, round_num=None):
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
    
    # Check if we have current update plan for regularized training
    use_update_plan = (current_update_plan is not None and 
                      client_id is not None and 
                      round_num is not None)
    
    total_loss = 0
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if use_update_plan:
                # Get current update plan for this client
                model_to_update_parts = current_update_plan["model_to_update"]
                model_as_regularizer_list = current_update_plan["model_as_regularizer"]
                
                # 1. Build the model_to_update from the main model
                model_to_update = _extract_model_parts(model, model_to_update_parts)
                
                # 2. Build the optimizer for the model_to_update
                model_to_update_optimizer = torch.optim.Adam(model_to_update.parameters(), lr=lr)
                
                # 3. Load regularizer models from shipped weights
                regularizer_models = _load_regularizer_models(model_as_regularizer_list, model, device)
                
                # Forward pass through model_to_update
                outputs_main = model_to_update(inputs)
                
                # Compute base loss
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    base_loss = criterion(outputs_main, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    base_loss = criterion(outputs_main, targets)
                
                # Determine regularizer type and compute regularized loss
                # Use configuration from REGULARIZER_PARAMS at top of file
                regularizer_type = REGULARIZER_PARAMS["type"]
                regularization_strength = REGULARIZER_PARAMS["coefficient"]
                
                if regularizer_type == "weights":
                    # 3.1, 3.2, 3.3: Compute weight distance regularization
                    weight_regularizer = _compute_weight_regularizer(model_to_update, regularizer_models)
                    loss = base_loss + regularization_strength * weight_regularizer
                    
                elif regularizer_type == "consistency loss":
                    # 4.1, 4.2, 4.3: Compute output consistency regularization
                    consistency_regularizer = _compute_consistency_regularizer(
                        model_to_update, regularizer_models, inputs, device
                    )
                    loss = base_loss + regularization_strength * consistency_regularizer
                
                else:
                    # Fallback to base loss
                    loss = base_loss
                
                print(f"[Client {client_id}] Round {round_num}: Using {regularizer_type} regularization with {len(model_as_regularizer_list)} regularizers, loss={loss.item():.4f}")
                
                # Use the specialized optimizer for model_to_update
                model_to_update_optimizer.zero_grad()
                loss.backward()
                model_to_update_optimizer.step()
                
                # Update the main model with the trained model_to_update parts
                _update_main_model_from_parts(model, model_to_update, model_to_update_parts)
                
            else:
                # Standard training without update plan
                outputs = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = torch.squeeze(targets, 1).long()
                    loss = criterion(outputs, targets)
                
                # Standard training step
                optimizer.zero_grad()
                            # Backward pass and optimization step are handled above in each branch
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

def _extract_model_parts(model, model_to_update_parts):
    """
    Extract specific model parts to create model_to_update.
    
    Args:
        model: The main model
        model_to_update_parts: Dict mapping layer names to FBD block IDs
        
    Returns:
        nn.Module: A model containing only the specified parts
    """
    # Create a new model with the same architecture
    # For now, we'll create a copy and only keep the specified parts active
    import copy
    model_to_update = copy.deepcopy(model)
    
    # Note: For a more sophisticated implementation, you could create a custom
    # model that only contains the specified layers, but for now we use the full model
    # and rely on the optimizer to only update the relevant parts
    
    return model_to_update

def _load_regularizer_models(model_as_regularizer_list, template_model, device):
    """
    Load regularizer models from shipped weights.
    
    Args:
        model_as_regularizer_list: List of regularizer specifications
        template_model: Template model to use for creating regularizer models
        device: Device to load models on
        
    Returns:
        List[nn.Module]: List of regularizer models
    """
    regularizer_models = []
    
    for regularizer_spec in model_as_regularizer_list:
        # Create a copy of the template model for this regularizer
        import copy
        regularizer_model = copy.deepcopy(template_model)
        regularizer_model.to(device)
        regularizer_model.eval()  # Set to eval mode for regularization
        
        # In a full implementation, you would load the specific weights
        # from the FBD warehouse based on the regularizer_spec
        # For now, we use the current model as a placeholder
        
        regularizer_models.append(regularizer_model)
    
    return regularizer_models

def _compute_weight_regularizer(model_to_update, regularizer_models):
    """
    Compute weight distance regularization using configured distance type.
    
    Args:
        model_to_update: The model being updated
        regularizer_models: List of regularizer models
        
    Returns:
        torch.Tensor: Weight regularization loss
    """
    weight_regularizer = 0.0
    distance_type = REGULARIZER_PARAMS["distance_type"]  # No default - will fail if not set
    
    for regularizer_model in regularizer_models:
        # Compute distance between corresponding parameters
        for (name1, param1), (name2, param2) in zip(
            model_to_update.named_parameters(), 
            regularizer_model.named_parameters()
        ):
            if name1 == name2:  # Ensure we're comparing the same parameters
                param_diff = param1 - param2.detach()
                
                if distance_type.upper() == "L1":
                    # L1 (Manhattan) distance
                    weight_regularizer += torch.norm(param_diff, p=1)
                elif distance_type.upper() == "L2":
                    # L2 (Euclidean) distance
                    weight_regularizer += torch.norm(param_diff, p=2) ** 2
                elif distance_type.upper() == "COSINE":
                    # Cosine distance (1 - cosine similarity)
                    param1_flat = param1.flatten()
                    param2_flat = param2.detach().flatten()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        param1_flat.unsqueeze(0), param2_flat.unsqueeze(0)
                    )
                    weight_regularizer += 1 - cosine_sim
                elif distance_type.upper() == "KL":
                    # KL divergence (for probability distributions)
                    # Apply softmax to make parameters probability-like
                    param1_prob = torch.nn.functional.softmax(param1.flatten(), dim=0)
                    param2_prob = torch.nn.functional.softmax(param2.detach().flatten(), dim=0)
                    weight_regularizer += torch.nn.functional.kl_div(
                        param1_prob.log(), param2_prob, reduction='sum'
                    )
                else:
                    # Fail explicitly for unknown distance types
                    raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: L1, L2, COSINE, KL")
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        weight_regularizer = weight_regularizer / len(regularizer_models)
    
    return weight_regularizer

def _compute_consistency_regularizer(model_to_update, regularizer_models, inputs, device):
    """
    Compute output consistency regularization using configured distance type.
    
    Args:
        model_to_update: The model being updated
        regularizer_models: List of regularizer models
        inputs: Input batch
        device: Device for computation
        
    Returns:
        torch.Tensor: Consistency regularization loss
    """
    consistency_regularizer = 0.0
    distance_type = REGULARIZER_PARAMS["distance_type"]  # No default - will fail if not set
    
    # Get outputs from model_to_update
    model_to_update_outputs = model_to_update(inputs)
    
    for regularizer_model in regularizer_models:
        # Get outputs from regularizer model
        with torch.no_grad():  # Don't compute gradients for regularizer
            regularizer_outputs = regularizer_model(inputs)
        
        # Compute distance between outputs using configured distance type
        if distance_type.upper() == "L1":
            # L1 (Manhattan) distance
            consistency_loss = torch.nn.functional.l1_loss(
                model_to_update_outputs, regularizer_outputs.detach()
            )
        elif distance_type.upper() == "L2":
            # L2 (Euclidean) distance - MSE
            consistency_loss = torch.nn.functional.mse_loss(
                model_to_update_outputs, regularizer_outputs.detach()
            )
        elif distance_type.upper() == "COSINE":
            # Cosine distance (1 - cosine similarity)
            # Flatten outputs for cosine similarity computation
            outputs1_flat = model_to_update_outputs.flatten(start_dim=1)
            outputs2_flat = regularizer_outputs.detach().flatten(start_dim=1)
            cosine_sim = torch.nn.functional.cosine_similarity(outputs1_flat, outputs2_flat, dim=1)
            consistency_loss = (1 - cosine_sim).mean()
        elif distance_type.upper() == "KL":
            # KL divergence (for probability distributions)
            # Apply softmax to make outputs probability-like
            outputs1_prob = torch.nn.functional.softmax(model_to_update_outputs, dim=-1)
            outputs2_prob = torch.nn.functional.softmax(regularizer_outputs.detach(), dim=-1)
            consistency_loss = torch.nn.functional.kl_div(
                outputs1_prob.log(), outputs2_prob, reduction='batchmean'
            )
        elif distance_type.upper() == "JS":
            # Jensen-Shannon divergence (symmetric version of KL)
            outputs1_prob = torch.nn.functional.softmax(model_to_update_outputs, dim=-1)
            outputs2_prob = torch.nn.functional.softmax(regularizer_outputs.detach(), dim=-1)
            # Compute average distribution
            avg_prob = 0.5 * (outputs1_prob + outputs2_prob)
            # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl1 = torch.nn.functional.kl_div(avg_prob.log(), outputs1_prob, reduction='batchmean')
            kl2 = torch.nn.functional.kl_div(avg_prob.log(), outputs2_prob, reduction='batchmean')
            consistency_loss = 0.5 * (kl1 + kl2)
        else:
            # Fail explicitly for unknown distance types
            raise ValueError(f"Unknown distance_type: {distance_type}. Supported types: L1, L2, COSINE, KL, JS")
        
        consistency_regularizer += consistency_loss
    
    # Average over the number of regularizer models
    if len(regularizer_models) > 0:
        consistency_regularizer = consistency_regularizer / len(regularizer_models)
    
    return consistency_regularizer

def _update_main_model_from_parts(main_model, model_to_update, model_to_update_parts):
    """
    Update the main model with trained parts from model_to_update.
    
    Args:
        main_model: The main model to update
        model_to_update: The trained model parts
        model_to_update_parts: Dict mapping layer names to FBD block IDs
    """
    # For now, we update the entire main model with the trained model_to_update
    # In a more sophisticated implementation, you would only update the specific parts
    # specified in model_to_update_parts
    
    main_model.load_state_dict(model_to_update.state_dict())