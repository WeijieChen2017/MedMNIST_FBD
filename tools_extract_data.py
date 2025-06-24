import json
import os
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt

directory = 'col5_fbd/bloodmnist_0621_0051'
output_csv_file = directory + '/bloodmnist_training_records.csv'

all_data = []

for i in range(101):
    file_path = os.path.join(directory, f'fbd_comprehensive_evaluation_round_{i}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            round_num = data.get('round')
            if data.get('success'):
                individual_results = data.get('individual_results', {})
                for model_name in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'Averaging']:
                    model_data = individual_results.get(model_name)
                    if model_data:
                        row = {
                            'round': round_num,
                            'model_name': model_name,
                            'loss': model_data.get('loss'),
                            'accuracy': model_data.get('accuracy'),
                            'auc': model_data.get('auc'),
                            'total_samples': model_data.get('total_samples'),
                            'total_batches': model_data.get('total_batches')
                        }
                        all_data.append(row)

if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_file, index=False)
    print(f"Data extracted and saved to {output_csv_file}")
else:
    print("No data was extracted. Please check the file paths and content.") 

# Set plot style
sns.set_style("whitegrid")

auc_savename = directory + '/bloodmnist_auc_vs_epoch.png'
accuracy_savename = directory + '/bloodmnist_accuracy_vs_epoch.png'
loss_savename = directory + '/bloodmnist_loss_vs_epoch.png'

# Plot AUC
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='auc', hue='model_name')
plt.title('AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(title='Model')
plt.savefig(auc_savename)
plt.close()

# Plot Accuracy
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='accuracy', hue='model_name')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(title='Model')
plt.savefig(accuracy_savename)
plt.close()

# Plot Loss
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='loss', hue='model_name')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(title='Model')
plt.savefig(loss_savename)
plt.close()

print(f"Plots saved as {auc_savename}, {accuracy_savename}, and {loss_savename}") 