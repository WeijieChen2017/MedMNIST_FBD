"""
Communication utilities for FBD Federated Learning
Handles weight transfer between server and clients
"""

import torch
import json
import os
import pickle
from typing import Dict, Any, List
import time

class WeightTransfer:
    """
    Handles serialization, transfer, and deserialization of model weights
    between server and clients. Uses file-based communication for testing.
    """
    
    def __init__(self, communication_dir="fbd_comm"):
        """
        Initialize weight transfer system.
        
        Args:
            communication_dir (str): Directory for communication files
        """
        self.comm_dir = communication_dir
        os.makedirs(self.comm_dir, exist_ok=True)
        
        # Create subdirectories for organized communication
        self.server_to_client_dir = os.path.join(self.comm_dir, "server_to_client")
        self.client_to_server_dir = os.path.join(self.comm_dir, "client_to_server")
        
        os.makedirs(self.server_to_client_dir, exist_ok=True)
        os.makedirs(self.client_to_server_dir, exist_ok=True)
    
    def serialize_weights(self, weights_dict: Dict[str, Dict[str, torch.Tensor]]) -> bytes:
        """
        Serialize weights dictionary to bytes for transfer.
        
        Args:
            weights_dict: Dictionary mapping layer names to state_dicts
            
        Returns:
            bytes: Serialized weights
        """
        return pickle.dumps(weights_dict)
    
    def deserialize_weights(self, weights_bytes: bytes) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Deserialize weights from bytes.
        
        Args:
            weights_bytes: Serialized weights
            
        Returns:
            Dict: Weights dictionary
        """
        return pickle.loads(weights_bytes)
    
    def server_send_weights(self, client_id: int, round_num: int, weights_dict: Dict[str, Dict[str, torch.Tensor]]):
        """
        Server sends weights to a specific client.
        
        Args:
            client_id: Target client ID
            round_num: Current round number
            weights_dict: Weights to send (mapping layer names to state_dicts)
        """
        # Create message with metadata
        message = {
            'round_num': round_num,
            'client_id': client_id,
            'timestamp': time.time(),
            'message_type': 'shipping_weights',
            'weights': weights_dict
        }
        
        # Save to file for client to pick up
        filename = f"round_{round_num}_client_{client_id}_shipping.pkl"
        filepath = os.path.join(self.server_to_client_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(message, f)
    
    def client_receive_weights(self, client_id: int, round_num: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Client receives weights from server.
        
        Args:
            client_id: This client's ID
            round_num: Current round number
            
        Returns:
            Dict: Received weights dictionary
        """
        filename = f"round_{round_num}_client_{client_id}_shipping.pkl"
        filepath = os.path.join(self.server_to_client_dir, filename)
        
        # Wait for file to exist (polling simulation)
        max_wait = 30  # seconds
        wait_time = 0
        while not os.path.exists(filepath) and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not os.path.exists(filepath):
            raise TimeoutError(f"No weights received for client {client_id} round {round_num}")
        
        with open(filepath, 'rb') as f:
            message = pickle.load(f)
        
        # Verify message is for this client and round
        assert message['client_id'] == client_id
        assert message['round_num'] == round_num
        assert message['message_type'] == 'shipping_weights'
        
        return message['weights']
    
    def client_send_weights(self, client_id: int, round_num: int, weights_dict: Dict[str, Dict[str, torch.Tensor]], request_list: List[str]):
        """
        Client sends requested weights back to server.
        
        Args:
            client_id: This client's ID
            round_num: Current round number
            weights_dict: Weights to send (mapping block IDs to state_dicts)
            request_list: List of block IDs that were requested
        """
        message = {
            'round_num': round_num,
            'client_id': client_id,
            'timestamp': time.time(),
            'message_type': 'request_response',
            'weights': weights_dict,
            'request_list': request_list
        }
        
        filename = f"round_{round_num}_client_{client_id}_response.pkl"
        filepath = os.path.join(self.client_to_server_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(message, f)
    
    def server_receive_weights(self, client_id: int, round_num: int) -> tuple:
        """
        Server receives weights from a specific client.
        
        Args:
            client_id: Source client ID
            round_num: Current round number
            
        Returns:
            tuple: (weights_dict, request_list)
        """
        filename = f"round_{round_num}_client_{client_id}_response.pkl"
        filepath = os.path.join(self.client_to_server_dir, filename)
        
        # Wait for file to exist
        max_wait = 30  # seconds
        wait_time = 0
        while not os.path.exists(filepath) and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not os.path.exists(filepath):
            raise TimeoutError(f"No response received from client {client_id} round {round_num}")
        
        with open(filepath, 'rb') as f:
            message = pickle.load(f)
        
        # Verify message
        assert message['client_id'] == client_id
        assert message['round_num'] == round_num
        assert message['message_type'] == 'request_response'
        
        return message['weights'], message['request_list']
    
    def server_send_request_list(self, client_id: int, round_num: int, request_list: List[str]):
        """
        Server sends request list to client.
        
        Args:
            client_id: Target client ID
            round_num: Current round number
            request_list: List of block IDs to request
        """
        message = {
            'round_num': round_num,
            'client_id': client_id,
            'timestamp': time.time(),
            'message_type': 'request_list',
            'request_list': request_list
        }
        
        filename = f"round_{round_num}_client_{client_id}_request.pkl"
        filepath = os.path.join(self.server_to_client_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(message, f)
    
    def client_receive_request_list(self, client_id: int, round_num: int) -> List[str]:
        """
        Client receives request list from server.
        
        Args:
            client_id: This client's ID
            round_num: Current round number
            
        Returns:
            List[str]: List of block IDs to send back
        """
        filename = f"round_{round_num}_client_{client_id}_request.pkl"
        filepath = os.path.join(self.server_to_client_dir, filename)
        
        # Wait for file to exist
        max_wait = 30  # seconds
        wait_time = 0
        while not os.path.exists(filepath) and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not os.path.exists(filepath):
            raise TimeoutError(f"No request list received for client {client_id} round {round_num}")
        
        with open(filepath, 'rb') as f:
            message = pickle.load(f)
        
        assert message['client_id'] == client_id
        assert message['round_num'] == round_num
        assert message['message_type'] == 'request_list'
        
        return message['request_list']
    
    def cleanup_round(self, round_num: int):
        """
        Clean up communication files for a specific round.
        
        Args:
            round_num: Round number to clean up
        """
        for directory in [self.server_to_client_dir, self.client_to_server_dir]:
            for filename in os.listdir(directory):
                if filename.startswith(f"round_{round_num}_"):
                    filepath = os.path.join(directory, filename)
                    os.remove(filepath)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get statistics about communication files.
        
        Returns:
            Dict: Communication statistics
        """
        stats = {
            'server_to_client_files': len(os.listdir(self.server_to_client_dir)),
            'client_to_server_files': len(os.listdir(self.client_to_server_dir)),
            'total_size_mb': 0
        }
        
        # Calculate total size
        total_size = 0
        for directory in [self.server_to_client_dir, self.client_to_server_dir]:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                total_size += os.path.getsize(filepath)
        
        stats['total_size_mb'] = total_size / (1024 * 1024)
        return stats 