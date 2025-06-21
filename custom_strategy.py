import os
import pandas as pd
from flwr.server.strategy import FedAvg, FedProx
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union, Dict, Optional
import pickle
from collections import OrderedDict
import torch

class CsvLoggingFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.output_dir = kwargs.pop("output_dir", ".")
        self.csv_path = os.path.join(self.output_dir, "results.csv")
        self.client_names = kwargs.pop("client_names", None)
        super().__init__(*args, **kwargs)
        self.round_history: Dict[int, Dict] = {}
        
        # If no client_names provided, we'll create default mapping when we see clients
        self._client_id_to_name: Dict[str, str] = {}
        self._next_default_id = 0
        self.fit_metrics_df = pd.DataFrame()
        self.eval_metrics_df = pd.DataFrame()

    def _get_client_name(self, client_id: str) -> str:
        """Get clean client name for a given client ID."""
        if self.client_names and client_id in self.client_names:
            return self.client_names[client_id]
        
        # If not in provided mapping, create default name
        if client_id not in self._client_id_to_name:
            self._client_id_to_name[client_id] = str(self._next_default_id)
            self._next_default_id += 1
        
        return self._client_id_to_name[client_id]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Check if clients are sending dictionaries
        first_client, first_res = results[0]
        if first_res.parameters.tensor_type == "dict":
            # Deserialize, aggregate, and serialize
            
            # 1. Deserialize all client weights
            deserialized_weights = []
            for client, fit_res in results:
                deserialized_weights.append(pickle.loads(fit_res.parameters.tensors[0]))

            # 2. Aggregate the weights
            aggregated_weights_dict = {}
            # Get the layer names from the first client (assuming all clients send same layers)
            layer_names = deserialized_weights[0].keys()

            for layer_name in layer_names:
                # Aggregate each tensor in the layer's state_dict
                layer_state_dict = deserialized_weights[0][layer_name]
                aggregated_state_dict = OrderedDict()

                for tensor_name in layer_state_dict.keys():
                    tensors_to_average = [client_weights[layer_name][tensor_name] for client_weights in deserialized_weights]
                    # Simple averaging
                    aggregated_tensor = torch.stack(tensors_to_average).mean(dim=0)
                    aggregated_state_dict[tensor_name] = aggregated_tensor
                
                aggregated_weights_dict[layer_name] = aggregated_state_dict
            
            # 3. Serialize the aggregated weights
            serialized_aggregated_weights = pickle.dumps(aggregated_weights_dict)
            parameters_aggregated = Parameters(tensors=[serialized_aggregated_weights], tensor_type="dict")

        else:
            # Fallback to default FedAvg if not using custom dict format
            parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)

        # Aggregate metrics
        metrics_aggregated = super().aggregate_fit(server_round, results, failures)[1]

        round_data = self.round_history.get(server_round, {"round": server_round})

        if metrics_aggregated:
            round_data.update({f"agg_{k}": v for k, v in metrics_aggregated.items()})

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_name = self._get_client_name(cid)
            for key, value in fit_res.metrics.items():
                round_data[f"client_{client_name}_{key}"] = value
        
        self.round_history[server_round] = round_data
        
        self.fit_metrics_df = pd.DataFrame(list(self.round_history.values()))
        
        cols = sorted([col for col in self.fit_metrics_df.columns if col != 'round'])
        cols.insert(0, 'round')
        self.fit_metrics_df = self.fit_metrics_df.reindex(columns=cols)

        # Format numerical columns to 6 decimal places
        for col in self.fit_metrics_df.columns:
            if col != 'round' and pd.api.types.is_numeric_dtype(self.fit_metrics_df[col]):
                self.fit_metrics_df[col] = self.fit_metrics_df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

        self.fit_metrics_df.to_csv(self.csv_path, index=False)
        
        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is not None:
            loss, metrics = eval_res
            round_data = self.round_history.get(server_round, {"round": server_round})
            round_data["centralized_loss"] = loss
            round_data.update({f"centralized_{k}": v for k, v in metrics.items()})
            self.round_history[server_round] = round_data
        
        if self.round_history:
            df = pd.DataFrame(list(self.round_history.values()))
            
            cols = sorted([col for col in df.columns if col != 'round'])
            cols.insert(0, 'round')
            df = df.reindex(columns=cols)

            # Format numerical columns to 6 decimal places
            for col in df.columns:
                if col != 'round' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

            df.to_csv(self.csv_path, index=False)
        
        self.eval_metrics_df = pd.DataFrame([{
            'round': server_round,
            'centralized_loss': loss,
            'centralized_loss': loss,
            **metrics
        }])
        self.eval_metrics_df.to_csv(os.path.join(self.output_dir, "evaluate_metrics.csv"), index=False)
        
        return eval_res 

class CsvLoggingFedProx(FedProx):
    def __init__(self, *args, **kwargs):
        # Extract FedProx-specific parameters
        self.mu = kwargs.pop("mu", 0.1)  # Proximal term weight (default: 0.1)
        
        # Extract logging parameters
        self.output_dir = kwargs.pop("output_dir", ".")
        self.csv_path = os.path.join(self.output_dir, "results.csv")
        self.client_names = kwargs.pop("client_names", None)
        
        # Initialize parent class with proximal_mu parameter (Flower's FedProx uses proximal_mu)
        super().__init__(*args, proximal_mu=self.mu, **kwargs)
        
        # Initialize logging
        self.round_history: Dict[int, Dict] = {}
        self._client_id_to_name: Dict[str, str] = {}
        self._next_default_id = 0
        self.fit_metrics_df = pd.DataFrame()
        self.eval_metrics_df = pd.DataFrame()

    def _get_client_name(self, client_id: str) -> str:
        """Get clean client name for a given client ID."""
        if self.client_names and client_id in self.client_names:
            return self.client_names[client_id]
        
        # If not in provided mapping, create default name
        if client_id not in self._client_id_to_name:
            self._client_id_to_name[client_id] = str(self._next_default_id)
            self._next_default_id += 1
        
        return self._client_id_to_name[client_id]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Check if clients are sending dictionaries
        if not results:
            return None, {}
        first_client, first_res = results[0]

        if first_res.parameters.tensor_type == "dict":
            # Deserialize, aggregate, and serialize
            deserialized_weights = []
            for client, fit_res in results:
                deserialized_weights.append(pickle.loads(fit_res.parameters.tensors[0]))

            aggregated_weights_dict = {}
            layer_names = deserialized_weights[0].keys()

            for layer_name in layer_names:
                layer_state_dict = deserialized_weights[0][layer_name]
                aggregated_state_dict = OrderedDict()

                for tensor_name in layer_state_dict.keys():
                    tensors_to_average = [client_weights[layer_name][tensor_name] for client_weights in deserialized_weights]
                    aggregated_tensor = torch.stack(tensors_to_average).mean(dim=0)
                    aggregated_state_dict[tensor_name] = aggregated_tensor
                
                aggregated_weights_dict[layer_name] = aggregated_state_dict
            
            serialized_aggregated_weights = pickle.dumps(aggregated_weights_dict)
            parameters_aggregated = Parameters(tensors=[serialized_aggregated_weights], tensor_type="dict")

        else:
            # Fallback to default FedProx if not using custom dict format
            parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)

        # Aggregate metrics
        metrics_aggregated = super().aggregate_fit(server_round, results, failures)[1]

        round_data = self.round_history.get(server_round, {"round": server_round})

        if metrics_aggregated:
            round_data.update({f"agg_{k}": v for k, v in metrics_aggregated.items()})

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_name = self._get_client_name(cid)
            for key, value in fit_res.metrics.items():
                round_data[f"client_{client_name}_{key}"] = value
        
        self.round_history[server_round] = round_data
        
        self.fit_metrics_df = pd.DataFrame(list(self.round_history.values()))
        
        cols = sorted([col for col in self.fit_metrics_df.columns if col != 'round'])
        cols.insert(0, 'round')
        self.fit_metrics_df = self.fit_metrics_df.reindex(columns=cols)

        # Format numerical columns to 6 decimal places
        for col in self.fit_metrics_df.columns:
            if col != 'round' and pd.api.types.is_numeric_dtype(self.fit_metrics_df[col]):
                self.fit_metrics_df[col] = self.fit_metrics_df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

        self.fit_metrics_df.to_csv(self.csv_path, index=False)
        
        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is not None:
            loss, metrics = eval_res
            round_data = self.round_history.get(server_round, {"round": server_round})
            round_data["centralized_loss"] = loss
            round_data.update({f"centralized_{k}": v for k, v in metrics.items()})
            self.round_history[server_round] = round_data
        
        if self.round_history:
            df = pd.DataFrame(list(self.round_history.values()))
            
            cols = sorted([col for col in df.columns if col != 'round'])
            cols.insert(0, 'round')
            df = df.reindex(columns=cols)

            # Format numerical columns to 6 decimal places
            for col in df.columns:
                if col != 'round' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

            df.to_csv(self.csv_path, index=False)
        
        self.eval_metrics_df = pd.DataFrame([{
            'round': server_round,
            'centralized_loss': loss,
            'centralized_loss': loss,
            **metrics
        }])
        self.eval_metrics_df.to_csv(os.path.join(self.output_dir, "evaluate_metrics.csv"), index=False)
        
        return eval_res 

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        # Log centralized evaluation results to a file
        round_data = self.round_history.get(server_round, {"round": server_round})
        round_data["centralized_loss"] = loss_aggregated
        round_data.update({f"centralized_{k}": v for k, v in metrics_aggregated.items()})
        self.round_history[server_round] = round_data
        
        if self.round_history:
            df = pd.DataFrame(list(self.round_history.values()))
            
            cols = sorted([col for col in df.columns if col != 'round'])
            cols.insert(0, 'round')
            df = df.reindex(columns=cols)

            # Format numerical columns to 6 decimal places
            for col in df.columns:
                if col != 'round' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

            df.to_csv(self.csv_path, index=False)
        
        self.eval_metrics_df = pd.DataFrame([{
            'round': server_round,
            'centralized_loss': loss_aggregated,
            'centralized_loss': loss_aggregated,
            **metrics_aggregated
        }])
        self.eval_metrics_df.to_csv(os.path.join(self.output_dir, "evaluate_metrics.csv"), index=False)
        
        return loss_aggregated, metrics_aggregated 