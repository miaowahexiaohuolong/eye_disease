import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file, extract training and validation loss for each epoch"""
    train_losses = []
    val_losses = []
    epochs = []
    
    try:
        # Read log file content
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regular expression patterns: match loss values in training and validation results
        train_pattern = r"Epoch \[(\d+)\] 训练结果 \| Loss: ([\d.]+) \| Precision:"
        val_pattern = r"Validation 结果 \| Loss: ([\d.]+) \| Precision:"
        
        train_matches = re.findall(train_pattern, content)
        val_matches = re.findall(val_pattern, content)
        
        # Ensure data integrity, use minimum length of matches
        num_epochs = min(len(train_matches), len(val_matches))
        
        for i in range(num_epochs):
            epoch_num = int(train_matches[i][0])
            train_loss = float(train_matches[i][1])
            val_loss = float(val_matches[i])
            
            epochs.append(epoch_num)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
    except Exception as e:
        print(f"Error parsing file {log_path}: {e}")
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses
    }

# Define log file paths (ensure these files are in current directory or specified path)
log_files = {
    'AlexNet': 'alexnet_train_20251129_110636.log',
    'DenseNet121': 'densenet121_train_20251129_002157.log',
    'ResNet50': 'resnet50_train_20251128_230925.log',
    'VGG16': 'vgg16_train_20251129_095914.log'
}

# Parse all log files
data = {}
for model_name, file_path in log_files.items():
    if Path(file_path).exists():
        data[model_name] = parse_log_file(file_path)
        print(f"✅ Parsed {model_name}: {len(data[model_name]['epochs'])} epochs")
    else:
        print(f"❌ File does not exist: {file_path}")

if not data:
    print("No log files found, please check file paths!")

# Create visualization charts - LINE PLOT with legend in upper left
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

# Assign colors for each model
colors = {
    'AlexNet': '#1f77b4',
    'DenseNet121': '#ff7f0e',
    'ResNet50': '#2ca02c',
    'VGG16': '#d62728'
}

for idx, (model_name, model_data) in enumerate(data.items()):
    ax = axes[idx]
    epochs = model_data['epochs']
    train_loss = model_data['train_loss']
    val_loss = model_data['val_loss']
    
    # Create line plots instead of bar charts
    ax.plot(epochs, train_loss, 
            label='Training Loss', 
            color=colors[model_name], 
            marker='o', markersize=4, 
            linewidth=1.5)
    ax.plot(epochs, val_loss, 
            label='Validation Loss', 
            color=colors[model_name], 
            marker='s', markersize=4, 
            linewidth=1.5, linestyle='--')
    
    # Set chart properties
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title(f'{model_name} Training and Validation Loss Changes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')  # Changed from 'upper right' to 'upper left'
    ax.grid(axis='both', alpha=0.3)
    
    # Add value labels if epoch count is small
    if len(epochs) <= 15:
        for i, (epoch, loss) in enumerate(zip(epochs, train_loss)):
            ax.annotate(f'{loss:.3f}', 
                        xy=(epoch, loss),
                        xytext=(0, 5), textcoords="offset points", 
                        ha='center', va='bottom', fontsize=7)
        
        for i, (epoch, loss) in enumerate(zip(epochs, val_loss)):
            ax.annotate(f'{loss:.3f}', 
                        xy=(epoch, loss),
                        xytext=(0, 5), textcoords="offset points", 
                        ha='center', va='bottom', fontsize=7)

plt.suptitle('Loss Comparison Across Different Models (Line Chart)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_loss_line_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary chart: final epoch loss comparison (STILL BAR CHART)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

model_names = list(data.keys())
final_train_losses = [data[model]['train_loss'][-1] for model in model_names]
final_val_losses = [data[model]['val_loss'][-1] for model in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width/2, final_train_losses, width, label='Final Training Loss', 
                color=[colors[name] for name in model_names], alpha=0.8)
bars2 = ax1.bar(x + width/2, final_val_losses, width, label='Final Validation Loss', 
                color=[colors[name] for name in model_names], alpha=0.8, hatch='//')

ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Loss Value', fontsize=12)
ax1.set_title('Final Epoch Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels for summary bar chart
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Create average loss comparison chart
avg_train_losses = [np.mean(data[model]['train_loss']) for model in model_names]
avg_val_losses = [np.mean(data[model]['val_loss']) for model in model_names]

bars3 = ax2.bar(x - width/2, avg_train_losses, width, label='Average Training Loss', 
                color=[colors[name] for name in model_names], alpha=0.8)
bars4 = ax2.bar(x + width/2, avg_val_losses, width, label='Average Validation Loss', 
                color=[colors[name] for name in model_names], alpha=0.8, hatch='//')

ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.set_title('Average Loss Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels for average loss bar chart
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_loss_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# Print loss comparison summary
print("\nLoss Comparison Summary:")
print("=" * 50)
for model_name in model_names:
    final_train = data[model_name]['train_loss'][-1]
    final_val = data[model_name]['val_loss'][-1]
    avg_train = np.mean(data[model_name]['train_loss'])
    avg_val = np.mean(data[model_name]['val_loss'])
    print(f"{model_name}:")
    print(f"  Final Training Loss: {final_train:.4f}")
    print(f"  Final Validation Loss: {final_val:.4f}")
    print(f"  Average Training Loss: {avg_train:.4f}")
    print(f"  Average Validation Loss: {avg_val:.4f}")
    print()