# Retinal Disease Classification (RFMiD) — Binary & Multiclass

This is a PyTorch project for retinal disease classification using the RFMiD dataset, supporting both binary (Normal vs Disease) and multiclass schemes. The project implements classic CNN model comparisons (AlexNet, ResNet50, DenseNet121, VGG16), and provides training, validation, testing workflows, logging, and best model saving.

## Training Parameter Configuration

| Parameter         | Binary Example   | Multiclass Example   | Description                |
|------------------|------------------|----------------------|----------------------------|
| batch_size       | 16               | 16                   | Samples per batch          |
| epochs           | 30               | 50                   | Training epochs            |
| lr               | 1e-4             | 1e-3                 | Learning rate              |
| img_size         | (224, 224)       | (224, 224)           | Input image size           |
| num_workers      | 12               | 12                   | DataLoader worker count    |
| optimizer        | AdamW            | AdamW                | Optimizer                  |
| loss function    | BCEWithLogitsLoss| CrossEntropyLoss     | Loss function              |
| device           | CUDA/CPU auto    | CUDA/CPU auto        | Training device            |
| output_dir       | model_comparison_results | multiclass_model_comparison | Output directory |
| models_to_train  | Multiple models  | Multiple models      | Models to compare          |

> Note: You can customize parameters in the CONFIG dictionary in `single_classification.py` and `multiclassification.py`.

# Retinal Disease Classification (RFMiD) — Binary & Multiclass

This is a PyTorch project for retinal disease classification using the RFMiD dataset, supporting both binary (Normal vs Disease) and multiclass schemes. The project implements classic CNN model comparisons (AlexNet, ResNet50, DenseNet121, VGG16, InceptionV3, SE-ResNet50, etc.), and provides training, validation, testing workflows, logging, and best model saving. You can find and download the dataset at:
https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification

## Dataset Download

## Main Contents (Overview)
- `single_classification.py`: Main script for binary classification training/evaluation (Normal vs Disease).
- `multiclassification.py`: Main script for multiclass training/evaluation (multiple disease categories).
- `dataset/`: Dataset structure (Training_Set, Evaluation_Set, Test_Set).
- `pictures/`: Data visualization for this dataset.
- `model_comparison_results/`: Binary classification outputs (logs, best_models).
- `multiclass_model_comparison/`: Multiclass classification outputs (logs, best_models).
- `rfmid_binary_model.pth`, `rfmid_multiclass_fixed.pth`: Generated after running the code once.

## Dataset Structure (Default Paths)
The project uses RFMiD-format CSV label files and image folders. Default paths can be modified in the `CONFIG` dictionary in the scripts:

- Training set: `dataset/Training_Set/Training`, labels: `dataset/Training_Set/RFMiD_Training_Labels.csv`
- Validation set: `dataset/Evaluation_Set/Validation`, labels: `dataset/Evaluation_Set/RFMiD_Validation_Labels.csv`
- Test set: `dataset/Test_Set/Test`, labels: `dataset/Test_Set/RFMiD_Testing_Labels.csv`

Make sure the CSV files contain `ID` (image ID) and `Disease_Risk` columns. For multiclass, the script expects binary disease columns from the third column onward. See code comments for label reading logic.

## Dependencies (Recommended Environment)
- Python 3.8+
- PyTorch (with matching torchvision version, CUDA 11+ recommended)
- torchvision
- numpy, pandas, pillow (PIL)
- scikit-learn
- efficientnet-pytorch (if using EfficientNet in `train.py`)

Example installation (Windows PowerShell):

```powershell
python -m pip install torch torchvision numpy pandas pillow scikit-learn efficientnet-pytorch
```

Note: Choose the correct torch/torchvision install command for your CUDA version (see pytorch.org).

## Quick Start (Example)

1. Prepare data: Place images in `dataset/Training_Set/Training` and other folders, ensure CSV label paths and columns are correct.
2. Modify configuration (optional): In `single_classification.py` (binary) or `with_valid_loss_mulit_classification.py` (multiclass), find the `CONFIG` dictionary and adjust paths, `img_size`, `batch_size`, `epochs`, `lr`, `models_to_train`, etc.
3. Run binary classification training:

```powershell
python train.py
```

This will train each model listed in `CONFIG['models_to_train']` and save logs and best models in `CONFIG['output_dir']` (default: `model_comparison_results`).

4. Run multiclass training:

```powershell
python with_valid_loss_mulit_classification.py
```

This will generate logs and best models in the `multiclass_model_comparison` directory.

5. Perform 5-fold cross-validation

In the `five_fold_comparsion` folder, there is a Python script that can be run to obtain the results of 5-fold computation.

## Configuration (Key Fields)
- `train_csv`, `train_img`, `val_csv`, `val_img`, `test_csv`, `test_img`: CSV and image folder paths.
- `img_size`: Input image size (default (224,224)). For InceptionV3, use (299,299).
- `batch_size`: Batch size, reduce if out of memory.
- `num_workers`: DataLoader worker count; on Windows, set to 0 or a small value to avoid issues.
- `epochs`, `lr`, `weight_decay`: Training epochs, learning rate, weight decay.
- `device`: Automatically selects CUDA (if available) or CPU.
- `output_dir`: Output directory for training results (logs, best_models, etc.).
- `models_to_train`: List of models to train/compare (e.g., `['alexnet','resnet50']`).

After modifying `CONFIG`, simply run the corresponding script to start training.

## Logs & Models
- Logs: Saved in `<output_dir>/logs/` for each model (includes training/validation metrics).
- Best models: Saved in `<output_dir>/best_models/` based on validation metrics (F1 in script), including model weights and optimizer state.

Example: After binary training, you will find `alexnet_best.pth` and others in `model_comparison_results/best_models/`.

## Notes & Suggestions
- Windows users: Large `num_workers` may cause DataLoader issues (try `num_workers=0` first).
- For InceptionV3, set `img_size` to (299,299) and pay attention to related warnings in the script.
- If out of memory, reduce `batch_size` or choose a smaller model (e.g., AlexNet).
- `efficientnet-pytorch`: To use EfficientNet, install the package and add the model to `CONFIG['models_to_train']` (already imported in the example).

## Training Results

### Binary Classification Model Comparison

| Model        | Precision | Recall | F1 |
|--------------|----------|--------|----|
| AlexNet      | 0.8994   | 0.9545 | 0.9262 |
| ResNet50     | 0.9423   | 0.9368 | 0.9395 |
| DenseNet121  | 0.9606   | 0.9150 | 0.9372 |
| VGG16        | 0.9545   | 0.9545 | 0.9545 |

### Multiclass Classification Model Comparison

| Model        | Precision | Recall | F1 |
|--------------|----------|--------|----|
| AlexNet      | 0.04     | 0.24   | 0.07 |
| ResNet50     | 0.59     | 0.62   | 0.59 |
| DenseNet121  | 0.59     | 0.60   | 0.57 |
| VGG16        | 0.04     | 0.21   | 0.07 |

### 5-Fold Cross-Validation: Binary Model Comparison

| Model        | F1    |
|--------------|-------|
| VGG16        | 0.9459 |
| ResNet50     | 0.9426 |
| EfficientNet | 0.8934 |

### 5-Fold Cross-Validation: Multiclass Model Comparison

| Model        | F1    |
|--------------|-------|
| VGG16        | 0.0123 |
| ResNet50     | 0.2038 |
| EfficientNet | 0.0165 |

---

## Results & Reproducibility
- Pretrained model files (`rfmid_binary_model.pth`, `rfmid_multiclass_fixed.pth`) are included in the repository and can be loaded for inference or evaluation.

## References
- For RFMiD dataset documentation and label format, see `Retinal_Disease_Classification_Binary_and_Multiclass_Perspectives.pdf` in the project.

## Contact & Contribution
Feel free to submit issues or pull requests to improve the scripts (e.g., add more models, training strategies, or visualizations).

---
# Retinal Disease Classification (RFMiD) — Binary & Multiclass

This is a PyTorch project for retinal disease classification using the RFMiD dataset, supporting both binary (Normal vs Disease) and multiclass schemes. The project implements classic CNN model comparisons (AlexNet, ResNet50, DenseNet121, VGG16), and provides training, validation, testing workflows, logging, and best model saving.

## Main Contents (Overview)
- `train.py`: Main script for binary classification training/evaluation (Normal vs Disease).
- `多分类.py`: Main script for multiclass training/evaluation (multiple disease categories).
- `dataset/`: Dataset structure (Training_Set, Evaluation_Set, Test_Set).
- `model_comparison_results/`: Binary classification outputs (logs, best_models).
- `multiclass_model_comparison/`: Multiclass classification outputs (logs, best_models).

## Dataset Structure (Default Paths)
The project uses RFMiD-format CSV label files and image folders. Default paths can be modified in the `CONFIG` dictionary in the scripts:

- Training set: `dataset/Training_Set/Training`, labels: `dataset/Training_Set/RFMiD_Training_Labels.csv`
- Validation set: `dataset/Evaluation_Set/Validation`, labels: `dataset/Evaluation_Set/RFMiD_Validation_Labels.csv`
- Test set: `dataset/Test_Set/Test`, labels: `dataset/Test_Set/RFMiD_Testing_Labels.csv`

Make sure the CSV files contain `ID` (image ID) and `Disease_Risk` columns. For multiclass, the script expects binary disease columns from the third column onward. See code comments for label reading logic.

## Dependencies (Recommended Environment)
- Python 3.8+
- PyTorch (with matching torchvision version, CUDA 11+ recommended)
- torchvision
- numpy, pandas, pillow (PIL)
- scikit-learn
- efficientnet-pytorch (if using EfficientNet in `train.py`)

Example installation (Windows PowerShell):

```powershell
python -m pip install torch torchvision numpy pandas pillow scikit-learn efficientnet-pytorch
```

Note: Choose the correct torch/torchvision install command for your CUDA version (see pytorch.org).

## Quick Start (Example)

1. Prepare data: Place images in `dataset/Training_Set/Training` and other folders, ensure CSV label paths and columns are correct.
2. Modify configuration (optional): In `train.py` (binary) or `多分类.py` (multiclass), find the `CONFIG` dictionary and adjust paths, `img_size`, `batch_size`, `epochs`, `lr`, `models_to_train`, etc.
3. Run binary classification training:

```powershell
python train.py
```

This will train each model listed in `CONFIG['models_to_train']` and save logs and best models in `CONFIG['output_dir']` (default: `model_comparison_results`).

4. Run multiclass training:

```powershell
python 多分类.py
```

This will generate logs and best models in the `multiclass_model_comparison` directory.

## Configuration (Key Fields)
- `train_csv`, `train_img`, `val_csv`, `val_img`, `test_csv`, `test_img`: CSV and image folder paths.
- `img_size`: Input image size (default (224,224)). For InceptionV3, use (299,299).
- `batch_size`: Batch size, reduce if out of memory.
- `num_workers`: DataLoader worker count; on Windows, set to 0 or a small value to avoid issues.
- `epochs`, `lr`, `weight_decay`: Training epochs, learning rate, weight decay.
- `device`: Automatically selects CUDA (if available) or CPU.
- `output_dir`: Output directory for training results (logs, best_models, etc.).
- `models_to_train`: List of models to train/compare (e.g., `['alexnet','resnet50']`).

After modifying `CONFIG`, simply run the corresponding script to start training.

## Logs & Models
- Logs: Saved in `<output_dir>/logs/` for each model (includes training/validation metrics).
- Best models: Saved in `<output_dir>/best_models/` based on validation metrics (F1 in script), including model weights and optimizer state.

Example: After binary training, you will find `alexnet_best.pth` and others in `model_comparison_results/best_models/`.

## Notes & Suggestions
- Windows users: Large `num_workers` may cause DataLoader issues (try `num_workers=0` first).
- If out of memory, reduce `batch_size` or choose a smaller model (e.g., AlexNet).
- `efficientnet-pytorch`: To use EfficientNet, install the package and add the model to `CONFIG['models_to_train']` (already imported in the example).

## Training Parameter Table

| Parameter      | Binary Example   | Multiclass Example   | Description                |
|----------------|------------------|----------------------|----------------------------|
| batch_size     | 16               | 16                   | Samples per batch          |
| epochs         | 30               | 50                   | Training epochs            |
| lr             | 1e-4             | 1e-3                 | Learning rate              |
| img_size       | (224, 224)       | (224, 224)           | Input image size           |
| num_workers    | 12               | 12                   | DataLoader worker count    |
| optimizer      | AdamW            | AdamW                | Optimizer                  |
| loss function  | BCEWithLogitsLoss| CrossEntropyLoss     | Loss function              |
| device         | CUDA/CPU auto    | CUDA/CPU auto        | Training device            |
| output_dir     | model_comparison_results | multiclass_model_comparison | Output directory |
| models_to_train| Multiple models  | Multiple models      | Models to compare          |

## Training Results

### Binary Classification Model Comparison

| Model        | Test Precision | Test Recall | Test F1 |
|--------------|---------------|-------------|---------|
| AlexNet      | 0.8994        | 0.9545      | 0.9262  |
| ResNet50     | 0.9423        | 0.9368      | 0.9395  |
| DenseNet121  | 0.9606        | 0.915       | 0.9372  |
| VGG16        | 0.9545        | 0.9545      | 0.9545  |

### Multiclass Classification Model Comparison

| Model        | Test Precision | Test Recall | Test F1 |
|--------------|---------------|-------------|---------|
| AlexNet      | 0.04          | 0.24        | 0.07    |
| ResNet50     | 0.59          | 0.62        | 0.59    |
| DenseNet121  | 0.59          | 0.60        | 0.57    |
| VGG16        | 0.04          | 0.21        | 0.07    |

---

## Contact & Contribution
Feel free to submit issues or pull requests to improve the scripts (e.g., add more models, training strategies, or visualizations).

---
