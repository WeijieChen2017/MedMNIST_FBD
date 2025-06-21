"""
Test script for FBD Logic
"""
import json
import os
from fbd_logic import generate_client_model_palettes, load_fbd_settings

def test_fbd_logic():
    """Test the FBD logic with the bloodmnist plan."""
    
    # Test parameters
    num_clients = 6
    fbd_file_path = "fbd_record/bloodmnist_info_1.py"
    
    print("=== Testing FBD Logic ===")
    print(f"Number of clients: {num_clients}")
    print(f"FBD file path: {fbd_file_path}")
    print()
    
    try:
        # First, let's examine the loaded settings
        fbd_trace, fbd_info, transparent_to_client = load_fbd_settings(fbd_file_path)
        
        print("=== FBD Configuration ===")
        print(f"TRANSPARENT_TO_CLIENT: {transparent_to_client}")
        print(f"FBD_INFO keys: {list(fbd_info.keys())}")
        print()
        
        print("=== FBD_TRACE Sample ===")
        print(f"Number of FBD_TRACE entries: {len(fbd_trace)}")
        print()
        
        # Generate client model palettes
        client_palettes = generate_client_model_palettes(num_clients, fbd_file_path)
        
        # Create fbd_test directory
        fbd_test_dir = "fbd_test"
        os.makedirs(fbd_test_dir, exist_ok=True)
        
        print("=== Client Model Palettes ===")
        for cid, palette in client_palettes.items():
            print(f"\nClient {cid}:")
            print(f"  Total FBD entries: {len(palette)}")
            
            # Group by model colors (if available)
            if transparent_to_client:
                colors = {}
                for fbd_id, entry in palette.items():
                    color = entry.get('color', 'Unknown')
                    if color not in colors:
                        colors[color] = []
                    colors[color].append(fbd_id)
                
                print(f"  Available models: {sorted(colors.keys())}")
            else:
                print(f"  Model colors hidden from client (transparent_to_client = False)")
            
            # Show entries by model part
            parts = {}
            for fbd_id, entry in palette.items():
                part = entry['model_part']
                if part not in parts:
                    parts[part] = []
                
                # Include color info only if transparent to client
                if transparent_to_client and 'color' in entry:
                    parts[part].append(f"{fbd_id}({entry['color']})")
                else:
                    parts[part].append(f"{fbd_id}")
            
            for part, entries in sorted(parts.items()):
                print(f"    {part}: {entries}")
            
            # Save each client's palette to a separate file
            client_file = os.path.join(fbd_test_dir, f"client_{cid}_palette.json")
            with open(client_file, "w") as f:
                json.dump(palette, f, indent=2)
            print(f"  ✓ Saved to {client_file}")
        
        print("\n=== Summary ===")
        print("✓ FBD logic test completed successfully!")
        
        # Save overall results to file for inspection
        with open("test_fbd_results.json", "w") as f:
            json.dump(client_palettes, f, indent=2)
        print("✓ Overall results saved to test_fbd_results.json")
        print(f"✓ Individual client palettes saved to {fbd_test_dir}/ directory")
        
        # List the created files
        print(f"\nFiles created in {fbd_test_dir}/:")
        for filename in sorted(os.listdir(fbd_test_dir)):
            file_path = os.path.join(fbd_test_dir, filename)
            file_size = os.path.getsize(file_path)
            print(f"  {filename} ({file_size} bytes)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fbd_logic() 