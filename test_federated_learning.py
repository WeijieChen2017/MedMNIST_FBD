"""
Comprehensive test for FBD Federated Learning System
Tests the complete workflow with server and multiple clients
"""

import os
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from fbd_server import FBDServer
from fbd_client import FBDClient

def run_client_simulation(client_id, 
                         fbd_config_path,
                         communication_dir,
                         rounds_to_run,
                         training_config):
    """
    Run a client simulation in a separate thread.
    
    Args:
        client_id: Client identifier
        fbd_config_path: Path to FBD configuration
        communication_dir: Communication directory
        rounds_to_run: List of rounds for this client to participate in
        training_config: Training configuration parameters
        
    Returns:
        Dict: Client results
    """
    try:
        # Initialize client
        client = FBDClient(
            client_id=client_id,
            fbd_config_path=fbd_config_path,
            communication_dir=communication_dir,
            num_classes=training_config['num_classes'],
            input_shape=training_config['input_shape'],
            learning_rate=training_config['learning_rate']
        )
        
        client_results = []
        
        # Run specified rounds
        for round_num in rounds_to_run:
            round_result = client.run_round(
                round_num=round_num,
                training_epochs=training_config['epochs_per_round'],
                batch_size=training_config['batch_size'],
                num_batches=training_config['num_batches'],
                verbose=training_config['verbose']
            )
            client_results.append(round_result)
            
            # Small delay between rounds
            time.sleep(0.1)
        
        # Final evaluation
        final_evaluation = client.evaluate_model()
        
        return {
            'client_id': client_id,
            'success': True,
            'round_results': client_results,
            'final_evaluation': final_evaluation,
            'training_history': client.get_training_history()
        }
        
    except Exception as e:
        return {
            'client_id': client_id,
            'success': False,
            'error': str(e)
        }

def test_single_round_federated_learning():
    """
    Test a single round of federated learning with all components.
    """
    print("=== Testing Single Round Federated Learning ===")
    
    # Configuration
    config = {
        'fbd_config_path': 'fbd_record/bloodmnist_info_1.py',
        'shipping_plan_path': 'fbd_record/shipping_plan.json',
        'request_plan_path': 'fbd_record/request_plan.json',
        'communication_dir': 'fbd_comm_test',
        'num_clients': 6,
        'num_classes': 8,
        'input_shape': (1, 28, 28),
        'test_round': 1
    }
    
    # Clean up previous test
    if os.path.exists(config['communication_dir']):
        import shutil
        shutil.rmtree(config['communication_dir'])
    
    # Initialize server
    server = FBDServer(
        fbd_config_path=config['fbd_config_path'],
        shipping_plan_path=config['shipping_plan_path'],
        request_plan_path=config['request_plan_path'],
        num_clients=config['num_clients'],
        communication_dir=config['communication_dir'],
        num_classes=config['num_classes'],
        input_shape=config['input_shape']
    )
    
    # Initialize clients
    clients = []
    for client_id in range(config['num_clients']):
        client = FBDClient(
            client_id=client_id,
            fbd_config_path=config['fbd_config_path'],
            communication_dir=config['communication_dir'],
            num_classes=config['num_classes'],
            input_shape=config['input_shape']
        )
        clients.append(client)
    
    print(f"\nRunning single round test (Round {config['test_round']})...")
    
    # Server: Execute shipping phase
    shipping_summary = server.shipping_phase(config['test_round'])
    
    # Clients: Receive weights and train
    client_results = []
    for client in clients:
        # Receive weights
        received_weights = client.receive_weights_from_server(config['test_round'])
        client.update_local_model(received_weights)
        
        # Local training
        training_result = client.perform_local_training(epochs=1, verbose=False)
        client_results.append(training_result)
        
        # Receive request and send back weights
        request_list = client.receive_request_list(config['test_round'])
        extracted_weights = client.extract_requested_weights(request_list)
        client.send_weights_to_server(config['test_round'], extracted_weights, request_list)
    
    # Server: Execute collection phase
    collection_summary = server.collection_phase(config['test_round'])
    
    # Server: Execute evaluation phase
    evaluation_results = server.evaluation_phase(config['test_round'])
    
    print(f"\n✅ Single Round Test Results:")
    print(f"  Shipped to {len(shipping_summary)} clients")
    print(f"  Collected from {len(collection_summary)} clients")
    print(f"  Evaluated {len(evaluation_results['models'])} models")
    
    return {
        'shipping_summary': shipping_summary,
        'collection_summary': collection_summary,
        'evaluation_results': evaluation_results,
        'client_results': client_results
    }

def test_multi_round_federated_learning(num_rounds=3):
    """
    Test multiple rounds of federated learning with concurrent clients.
    """
    print(f"\n=== Testing Multi-Round Federated Learning ({num_rounds} rounds) ===")
    
    # Configuration
    config = {
        'fbd_config_path': 'fbd_record/bloodmnist_info_1.py',
        'shipping_plan_path': 'fbd_record/shipping_plan.json',
        'request_plan_path': 'fbd_record/request_plan.json',
        'communication_dir': 'fbd_comm_multitest',
        'num_clients': 6,
        'num_classes': 8,
        'input_shape': (1, 28, 28)
    }
    
    training_config = {
        'epochs_per_round': 1,
        'batch_size': 32,
        'num_batches': 5,
        'learning_rate': 0.001,
        'verbose': False,
        'num_classes': config['num_classes'],
        'input_shape': config['input_shape']
    }
    
    # Clean up previous test
    if os.path.exists(config['communication_dir']):
        import shutil
        shutil.rmtree(config['communication_dir'])
    
    # Initialize server
    server = FBDServer(
        fbd_config_path=config['fbd_config_path'],
        shipping_plan_path=config['shipping_plan_path'],
        request_plan_path=config['request_plan_path'],
        num_clients=config['num_clients'],
        communication_dir=config['communication_dir'],
        num_classes=config['num_classes'],
        input_shape=config['input_shape']
    )
    
    all_results = []
    
    # Run multiple rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # Server: Shipping phase
        shipping_summary = server.shipping_phase(round_num)
        
        # Concurrent client execution
        with ThreadPoolExecutor(max_workers=config['num_clients']) as executor:
            futures = []
            
            for client_id in range(config['num_clients']):
                future = executor.submit(
                    run_client_simulation,
                    client_id,
                    config['fbd_config_path'],
                    config['communication_dir'],
                    [round_num],  # Single round
                    training_config
                )
                futures.append(future)
            
            # Wait for all clients to complete
            client_results = []
            for future in as_completed(futures):
                result = future.result()
                client_results.append(result)
                if result['success']:
                    print(f"  Client {result['client_id']}: Training completed")
                else:
                    print(f"  Client {result['client_id']}: Error - {result['error']}")
        
        # Server: Collection phase
        collection_summary = server.collection_phase(round_num)
        
        # Server: Evaluation phase
        evaluation_results = server.evaluation_phase(round_num)
        
        round_result = {
            'round': round_num,
            'shipping_summary': shipping_summary,
            'collection_summary': collection_summary,
            'evaluation_results': evaluation_results,
            'client_results': client_results
        }
        
        all_results.append(round_result)
        
        print(f"Round {round_num} completed:")
        print(f"  Successful clients: {sum(1 for r in client_results if r['success'])}")
        print(f"  Models evaluated: {len(evaluation_results['models'])}")
    
    return all_results

def test_server_only_simulation(num_rounds=5):
    """
    Test server-only simulation (no actual clients, server manages everything).
    """
    print(f"\n=== Testing Server-Only Simulation ({num_rounds} rounds) ===")
    
    # Configuration
    config = {
        'fbd_config_path': 'fbd_record/bloodmnist_info_1.py',
        'shipping_plan_path': 'fbd_record/shipping_plan.json',
        'request_plan_path': 'fbd_record/request_plan.json',
        'communication_dir': 'fbd_comm_server_only',
        'num_clients': 6,
        'num_classes': 8,
        'input_shape': (1, 28, 28)
    }
    
    # Initialize server
    server = FBDServer(
        fbd_config_path=config['fbd_config_path'],
        shipping_plan_path=config['shipping_plan_path'],
        request_plan_path=config['request_plan_path'],
        num_clients=config['num_clients'],
        communication_dir=config['communication_dir'],
        num_classes=config['num_classes'],
        input_shape=config['input_shape']
    )
    
    # Run federated learning simulation (server manages everything)
    results = server.run_federated_learning(
        start_round=1,
        end_round=num_rounds,
        verbose=True
    )
    
    print(f"\n✅ Server-Only Simulation Results:")
    print(f"  Completed rounds: {len(results)}")
    print(f"  Total time: {sum(r['round_time'] for r in results):.2f}s")
    print(f"  Average models evaluated per round: {sum(len(r['evaluation_results']['models']) for r in results) / len(results):.1f}")
    
    return results

def save_test_results(test_results, output_dir='fbd_test/federated_results'):
    """
    Save comprehensive test results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each test result
    for test_name, results in test_results.items():
        filename = f"{test_name}_results.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Saved {test_name} results to {filepath}")

def main():
    """
    Run comprehensive federated learning tests.
    """
    print("🚀 Starting Comprehensive FBD Federated Learning Tests")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Single round with all components
        print("\n📋 Test 1: Single Round Integration")
        test_results['single_round'] = test_single_round_federated_learning()
        print("✅ Single round test completed")
        
        # Test 2: Multi-round with concurrent clients
        print("\n📋 Test 2: Multi-Round Concurrent Clients")
        test_results['multi_round'] = test_multi_round_federated_learning(num_rounds=3)
        print("✅ Multi-round test completed")
        
        # Test 3: Server-only simulation
        print("\n📋 Test 3: Server-Only Simulation")
        test_results['server_only'] = test_server_only_simulation(num_rounds=5)
        print("✅ Server-only simulation completed")
        
        # Save all results
        print("\n💾 Saving test results...")
        save_test_results(test_results)
        
        # Print summary
        print("\n🎉 All Tests Completed Successfully!")
        print("\nTest Summary:")
        print(f"  ✅ Single round integration: {len(test_results['single_round']['client_results'])} clients")
        print(f"  ✅ Multi-round concurrent: {len(test_results['multi_round'])} rounds")
        print(f"  ✅ Server-only simulation: {len(test_results['server_only'])} rounds")
        print("\nResults saved to fbd_test/federated_results/")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 