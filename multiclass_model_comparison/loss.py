import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """解析日志文件，提取训练和验证损失"""
    train_losses = []
    val_losses = []
    epochs = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 提取训练损失（匹配中文格式）
        # 示例: "Epoch [1] 训练损失：2.7383"
        train_pattern = r"Epoch \[(\d+)\] 训练损失：([\d.]+)"
        train_matches = re.findall(train_pattern, content)
        
        # 2. 提取验证损失（如果存在）
        # 示例: "Epoch [1] 验证损失：2.1234"
        val_pattern = r"Epoch \[(\d+)\] 验证损失：([\d.]+)"
        val_matches = re.findall(val_pattern, content)
        
        # 创建验证损失字典，便于匹配
        val_dict = {int(epoch): float(loss) for epoch, loss in val_matches}
        
        for match in train_matches:
            epoch_num = int(match[0])
            train_loss = float(match[1])
            
            # 只添加有对应验证损失的轮次（如果验证损失存在）
            if epoch_num in val_dict:
                epochs.append(epoch_num)
                train_losses.append(train_loss)
                val_losses.append(val_dict[epoch_num])
            else:
                # 如果没有验证损失，只记录训练损失
                epochs.append(epoch_num)
                train_losses.append(train_loss)
                
    except Exception as e:
        print(f"❌ 解析文件 {log_path} 时出错: {e}")
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses  # 可能为空列表
    }

# 定义日志文件路径
log_files = {
    'AlexNet': 'alexnet_multiclass_train_20251129_173521.log',
    'DenseNet121': 'densenet121_multiclass_train_20251129_143526.log',
    'ResNet50': 'resnet50_multiclass_train_20251129_131548.log',
    'VGG16': 'vgg16_multiclass_train_20251129_160206.log'
}

# 解析所有日志文件
data = {}
for model_name, file_path in log_files.items():
    if Path(file_path).exists():
        parsed_data = parse_log_file(file_path)
        if parsed_data['train_loss']:
            data[model_name] = parsed_data
            val_info = f" (验证损失: {len(parsed_data['val_loss'])} 轮)" if parsed_data['val_loss'] else " (无验证损失)"
            print(f"✅ 解析 {model_name}: {len(parsed_data['epochs'])} 轮{val_info}")
        else:
            print(f"⚠️  解析 {model_name} 但未找到训练损失数据")
    else:
        print(f"❌ 文件不存在: {file_path}")

if not data:
    print("未找到日志文件或解析失败！")
    exit()

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors = {
    'AlexNet': '#1f77b4',
    'DenseNet121': '#ff7f0e',
    'ResNet50': '#2ca02c',
    'VGG16': '#d62728'
}

available_models = list(data.keys())

for idx, model_name in enumerate(available_models):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    epochs = data[model_name]['epochs']
    train_loss = data[model_name]['train_loss']
    val_loss = data[model_name]['val_loss']
    
    # 绘制训练损失
    ax.plot(epochs, train_loss, label='训练损失', 
            color=colors[model_name], marker='o', markersize=4, linewidth=2)
    
    # 如果存在验证损失，也绘制
    if val_loss:
        ax.plot(epochs, val_loss, label='验证损失', 
                color=colors[model_name], marker='s', markersize=4, 
                linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title(f'{model_name} 训练与验证损失曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    # 添加数值标签
    if len(epochs) <= 20:
        for epoch, loss in zip(epochs, train_loss):
            ax.annotate(f'{loss:.3f}', xy=(epoch, loss), xytext=(0, 5), 
                       textcoords="offset points", ha='center', va='bottom', fontsize=7)

# 隐藏空的子图
for idx in range(len(available_models), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('不同模型的损失对比 (需添加验证损失记录)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建摘要图表
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

model_names = available_models
final_train_losses = [data[model]['train_loss'][-1] for model in model_names]
avg_train_losses = [np.mean(data[model]['train_loss']) for model in model_names]

x = np.arange(len(model_names))
width = 0.35

# 最终损失对比
bars1 = ax1.bar(x - width/2, final_train_losses, width, label='最终训练损失', 
                color=[colors[name] for name in model_names], alpha=0.8)
bars2 = ax1.bar(x + width/2, avg_train_losses, width, label='平均训练损失', 
                color=[colors[name] for name in model_names], alpha=0.4)

ax1.set_xlabel('模型', fontsize=12)
ax1.set_ylabel('损失值', fontsize=12)
ax1.set_title('训练损失摘要', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=9)

# 验证损失对比（如果有数据）
has_val_data = any(len(data[model]['val_loss']) > 0 for model in model_names)
if has_val_data:
    final_val_losses = [data[model]['val_loss'][-1] if data[model]['val_loss'] else 0 
                        for model in model_names]
    avg_val_losses = [np.mean(data[model]['val_loss']) if data[model]['val_loss'] else 0 
                      for model in model_names]
    
    bars3 = ax2.bar(x - width/2, final_val_losses, width, label='最终验证损失', 
                    color=[colors[name] for name in model_names], alpha=0.8, hatch='//')
    bars4 = ax2.bar(x + width/2, avg_val_losses, width, label='平均验证损失', 
                    color=[colors[name] for name in model_names], alpha=0.4, hatch='//')
    
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.set_title('验证损失摘要（如可用）', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
else:
    ax2.text(0.5, 0.5, '验证损失数据未找到\n\n请在训练脚本中添加：\nlogging.info(f"Epoch [{epoch}] 验证损失：{val_loss:.4f}")', 
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

plt.tight_layout()
plt.savefig('loss_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印摘要
print("\n" + "="*60)
print("训练损失摘要")
print("="*60)
for model_name in model_names:
    train_losses = data[model_name]['train_loss']
    val_losses = data[model_name]['val_loss']
    
    print(f"\n{model_name}:")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  平均训练损失: {np.mean(train_losses):.4f}")
    print(f"  最低训练损失: {min(train_losses):.4f}")
    
    if val_losses:
        print(f"  最终验证损失: {val_losses[-1]:.4f}")
        print(f"  平均验证损失: {np.mean(val_losses):.4f}")
    else:
        print("  验证损失: 未记录")