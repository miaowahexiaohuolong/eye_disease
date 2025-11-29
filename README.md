# 视网膜疾病分类（RFMiD） — 二分类 & 多分类

这是一个用于视网膜疾病（RFMiD 数据集）分类的 PyTorch 项目，包含二分类（Normal vs Disease）和多分类方案。项目实现了常见的经典 CNN 模型对比（AlexNet、ResNet50、DenseNet121、VGG16），并提供训练、验证与测试流程、日志记录和最优模型保存。

## 主要内容（概要）
- `train.py`：二分类训练/评估主脚本（Normal vs Disease）。
- `多分类.py`：多分类训练/评估主脚本（多疾病类别分类）。
- `dataset/`：数据集结构（Training_Set、Evaluation_Set、Test_Set）。
- `model_comparison_results/`：二分类训练输出（logs、best_models）。
- `multiclass_model_comparison/`：多分类训练输出（logs、best_models）。

## 数据集结构（项目内约定路径）
项目默认使用 RFMiD 格式的 CSV 标签和图片文件夹，默认路径可以在脚本的 `CONFIG` 字典中修改：

- 训练集：`dataset/Training_Set/Training`，标签：`dataset/Training_Set/RFMiD_Training_Labels.csv`
- 验证集：`dataset/Evaluation_Set/Validation`，标签：`dataset/Evaluation_Set/RFMiD_Validation_Labels.csv`
- 测试集：`dataset/Test_Set/Test`，标签：`dataset/Test_Set/RFMiD_Testing_Labels.csv`

确保 CSV 中包含 `ID`（图片 ID）和 `Disease_Risk` 等列（多分类脚本期望 CSV 第三列及之后为疾病二值列），脚本中有注释说明标签读取逻辑。

## 依赖（建议环境）
- Python 3.8+
- PyTorch（与 torchvision 匹配的版本，建议使用 CUDA 11+ 对应的预编译包）
- torchvision
- numpy, pandas, pillow (PIL)
- scikit-learn
- efficientnet-pytorch（若在 `train.py` 中使用 EfficientNet）

示例安装（在 Windows PowerShell 中运行）：

```powershell
python -m pip install torch torchvision numpy pandas pillow scikit-learn efficientnet-pytorch
```

注意：根据你的 CUDA 版本选择合适的 torch/torchvision 安装命令（参见 pytorch.org）。

## 快速开始（示例）

1. 准备数据：把图片放到 `dataset/Training_Set/Training` 等对应目录，确保 CSV 标签文件路径和文件列格式正确。
2. 修改配置（可选）：在 `train.py`（二分类）或 `多分类.py`（多分类）中找到 `CONFIG` 字典，调整路径、`img_size`、`batch_size`、`epochs`、`lr`、`models_to_train` 等参数。
3. 运行二分类训练：

```powershell
python train.py
```

这会按 `CONFIG['models_to_train']` 中列出的模型逐个训练，并在 `CONFIG['output_dir']` 下保存日志与最优模型（默认：`model_comparison_results`）。

4. 运行多分类训练：

```powershell
python 多分类.py
```

这会在 `multiclass_model_comparison` 目录下生成日志和最佳模型。

## 配置说明（主要字段）
- `train_csv`, `train_img`, `val_csv`, `val_img`, `test_csv`, `test_img`：CSV 与图片目录路径。
- `img_size`：输入图片大小（(224,224) 默认）。若使用 InceptionV3，建议改为 (299,299)。
- `batch_size`：批次大小，显存不足可减小。
- `num_workers`：DataLoader 的 worker 数；Windows 上可适当设为 0 或较小值以避免问题。
- `epochs`, `lr`, `weight_decay`：训练轮次、学习率和权重衰减。
- `device`：默认会自动选择 CUDA（若可用）或 CPU。
- `output_dir`：训练输出目录（logs、best_models 等）。
- `models_to_train`：要训练/对比的模型列表（例如 `['alexnet','resnet50']`）。

修改 `CONFIG` 后直接运行对应脚本即可开始训练。

## 日志与模型
- 日志：`<output_dir>/logs/` 下会保存每个模型的训练日志（包含训练/验证过程的指标）。
- 最佳模型：`<output_dir>/best_models/` 保存按验证指标（脚本内为 F1）选取的最佳 checkpoint（包含模型权重和优化器状态）。

示例：默认二分类训练结束后，你会在 `model_comparison_results/best_models/` 找到 `alexnet_best.pth` 等文件。

## 注意事项与建议
- Windows 用户：`num_workers` 过大可能导致 DataLoader 问题（建议先用 `num_workers=0` 验证）。
- 当使用 InceptionV3 时，请将 `img_size` 改为 (299,299) 并留意脚本中有关 `transform_input` 的警告。
- 若显存不足，减少 `batch_size` 或选择更小模型（例如 AlexNet）。
- `efficientnet-pytorch`：若想在代码中启用 EfficientNet，请安装对应包并在 `CONFIG['models_to_train']` 中添加相应条目（示例中已 import）。


## 训练成果展示（下面为示例）

### 二分类模型对比结果

| 模型         | 验证集 Precision | 验证集 Recall | 验证集 F1 | 测试集 Precision | 测试集 Recall | 测试集 F1 |
|--------------|------------------|--------------|-----------|------------------|--------------|-----------|
| AlexNet      | 0.92             | 0.90         | 0.91      | 0.91             | 0.89         | 0.90      |
| ResNet50     | 0.93             | 0.91         | 0.92      | 0.92             | 0.90         | 0.91      |
| DenseNet121  | 0.94             | 0.92         | 0.93      | 0.93             | 0.91         | 0.92      |
| VGG16        | 0.91             | 0.89         | 0.90      | 0.90             | 0.88         | 0.89      |



### 多分类模型对比结果

| 模型         | 验证集 Accuracy | 验证集 Macro F1 | 测试集 Accuracy | 测试集 Macro F1 |
|--------------|-----------------|-----------------|-----------------|-----------------|
| AlexNet      | 0.85            | 0.82            | 0.84            | 0.81            |
| ResNet50     | 0.87            | 0.84            | 0.86            | 0.83            |
| DenseNet121  | 0.88            | 0.85            | 0.87            | 0.84            |
| VGG16        | 0.83            | 0.80            | 0.82            | 0.79            |


> 注：Macro F1 为所有类别 F1 的平均值，具体每类 F1 可在日志文件中查阅。

---

## 联系 & 贡献
欢迎提交 issue 或 pull request 来改进脚本（例如：增加更多模型、训练策略或可视化）。

---

