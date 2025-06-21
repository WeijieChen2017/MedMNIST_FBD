import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
csv_file = 'bloodmnist_training_records.csv'
df = pd.read_csv(csv_file)

# Set plot style
sns.set_style("whitegrid")

# Plot AUC
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='auc', hue='model_name')
plt.title('AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(title='Model')
plt.savefig('bloodmnist_auc_vs_epoch.png')
plt.close()

# Plot Accuracy
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='accuracy', hue='model_name')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(title='Model')
plt.savefig('bloodmnist_accuracy_vs_epoch.png')
plt.close()

# Plot Loss
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='round', y='loss', hue='model_name')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(title='Model')
plt.savefig('bloodmnist_loss_vs_epoch.png')
plt.close()

print("Plots saved as bloodmnist_auc_vs_epoch.png, bloodmnist_accuracy_vs_epoch.png, and bloodmnist_loss_vs_epoch.png") 