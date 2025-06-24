folder = "col3_FedAvg_2d" 
# folder = "col4_train_FedProx"

DATASETS_2D = [
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
]

import glob
import json
import pandas as pd

# Find all *_eval.json files recursively in col2_vanilla_2d
result_json_list = sorted(glob.glob(f"{folder}/**/results.csv", recursive=True))
print(f"All evaluation JSON files found in {folder}:")
print("=" * 60)
for json_path in result_json_list:
    print(json_path)

print(f"\nTotal files found: {len(result_json_list)}")

# Extract metrics from all CSV files
data_rows = []

print("\nExtracting metrics from CSV files...")
print("=" * 60)

for csv_path in result_json_list:
    try:
        # Parse path components
        dataset = csv_path.split("/")[1]
        model = csv_path.split("/")[2]
        image_size = csv_path.split("/")[3]
        model_index = csv_path.split("/")[4]
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Find the round with the best centralized test accuracy
        best_round_idx = df['centralized_test_acc'].idxmax()
        best_round = df.iloc[best_round_idx]
        
        # Extract centralized metrics from the best round
        row = {
            'dataset': dataset,
            'model': model,
            'image_size': image_size,
            'model_index': model_index,
            'best_round': best_round['round'],
            # 'centralized_loss': best_round['centralized_loss'],
            'centralized_test_auc': best_round['centralized_test_auc'],
            'centralized_test_acc': best_round['centralized_test_acc'],
            # 'centralized_test_loss': best_round['centralized_test_loss'],
            'file_path': csv_path
        }
        data_rows.append(row)
        print(f"✓ {csv_path} -> Dataset: {dataset}, Model: {model}, Best Round: {best_round['round']}, Best Test Acc: {best_round['centralized_test_acc']:.4f}")
        
    except Exception as e:
        print(f"✗ Error reading {csv_path}: {e}")

# Create DataFrame and save to CSV
if data_rows:
    df = pd.DataFrame(data_rows)
    
    # Sort by dataset, model, and model_index for better organization
    df = df.sort_values(['dataset', 'model', 'model_index'])
    
    # Save to CSV
    output_file = f"{folder}_centralized_metrics.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully extracted centralized metrics from {len(data_rows)} files")
    print(f"✓ Results saved to: {output_file}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"- Datasets: {df['dataset'].nunique()} unique ({', '.join(sorted(df['dataset'].unique()))})")
    print(f"- Models: {df['model'].nunique()} unique ({', '.join(sorted(df['model'].unique()))})")
    print(f"- Average Best Test Acc: {df['centralized_test_acc'].mean():.4f} (±{df['centralized_test_acc'].std():.4f})")
    print(f"- Best Test Acc: {df['centralized_test_acc'].max():.4f}")
    print(f"- Worst Test Acc: {df['centralized_test_acc'].min():.4f}")
    print(f"- Average Best Round: {df['best_round'].mean():.1f}")
    
else:
    print("No data extracted!")

# col2_vanilla_2d/bloodmnist/resnet18/28/1/resnet18_28_1_eval.json