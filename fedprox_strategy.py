import os
import pandas as pd
import torch
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union, Dict, Optional

class CsvLoggingFedProx(FedAvg):
    """
    FedProx strategy with CSV logging.
    
    FedProx adds a proximal term to the local objective function to prevent
    client models from drifting too far from the global model.
    """
    def __init__(self, *args, **kwargs):
        # Extract FedProx-specific parameters
        self.mu = kwargs.pop("mu", 0.1)  # Proximal term weight (default: 0.1)
        
        # Extract logging parameters
        self.output_dir = kwargs.pop("output_dir", ".")
        self.csv_path = os.path.join(self.output_dir, "results.csv")
        self.client_names = kwargs.pop("client_names", None)
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Initialize logging
        self.round_history: Dict[int, Dict] = {}
        self._client_id_to_name: Dict[str, str] = {}
        self._next_default_id = 0
        
        # Store global model parameters for proximal term
        self.global_parameters = None

    def _get_client_name(self, client_id: str) -> str:
        """Get clean client name for a given client ID."""
        if self.client_names and client_id in self.client_names:
            return self.client_names[client_id]
        
        # If not in provided mapping, create default name
        if client_id not in self._client_id_to_name:
            self._client_id_to_name[client_id] = str(self._next_default_id)
            self._next_default_id += 1
        
        return self._client_id_to_name[client_id]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of training."""
        # Store global parameters for proximal term
        self.global_parameters = parameters
        
        # Get standard fit configuration
        fit_config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add proximal term weight to each client's config
        new_fit_config = []
        for client_proxy, fit_ins in fit_config:
            # Create a new config dictionary with FedProx parameters
            new_config = dict(fit_ins.config)
            new_config["mu"] = self.mu
            new_config["global_parameters"] = parameters_to_ndarrays(parameters)
            new_fit_config.append((client_proxy, new_config))
        
        return new_fit_config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and update global model."""
        if not results:
            return None, {}
        
        # Aggregate parameters using standard FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Update global parameters for next round
        if aggregated_parameters is not None:
            self.global_parameters = aggregated_parameters
        
        # Log metrics
        round_data = self.round_history.get(server_round, {"round": server_round})

        if aggregated_metrics:
            round_data.update({f"agg_{k}": v for k, v in aggregated_metrics.items()})

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_name = self._get_client_name(cid)
            for key, value in fit_res.metrics.items():
                round_data[f"client_{client_name}_{key}"] = value
        
        self.round_history[server_round] = round_data
        
        return aggregated_parameters, aggregated_metrics

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
        
        return eval_res 