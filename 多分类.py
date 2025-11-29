import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# ======================== 1. 核心配置（支持多模型+多分类） ========================
CONFIG = {
    # 路径配置（保持你的原始路径）
    'train_csv': 'E:\\eyes__disease\\dataset\\Training_Set\\RFMiD_Training_Labels.csv',
    'train_img': 'E:\\eyes__disease\\dataset\\Training_Set\\Training',
    'val_csv': 'E:\\eyes__disease\\dataset\\Evaluation_Set\\RFMiD_Validation_Labels.csv',
    'val_img': 'E:\\eyes__disease\\dataset\\Evaluation_Set\\Validation',
    'test_csv': 'E:\\eyes__disease\\dataset\\Test_Set\\RFMiD_Testing_Labels.csv',
    'test_img': 'E:\\eyes__disease\\dataset\\Test_Set\\Test',
    
    # 多分类相关配置
    'all_class_names': [],  # CSV第三列开始的所有列（含无样本类别）
    'actual_class_names': [],  # 实际有样本的类别（含Normal）
    'img_ext': '.png',          
    'img_size': (224, 224),    # InceptionV3建议改为(299,299)，其他模型兼容224
    'batch_size': 16,          # 多模型适配（VGG16/InceptionV3建议8；显存≥12G用16）
    'num_workers': 12,          
    'epochs': 50,              
    'lr': 1e-3,                # InceptionV3/SE-ResNet50建议5e-5
    'weight_decay': 1e-5,       
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'multiclass_model_comparison',  # 多模型结果保存根目录
    'models_to_train': [        # 可选择训练的多分类模型（可增删）
        #'lenet5',
        'alexnet',
        #'vgg16',
        #'inception_v3',
        #'resnet50',
        #'densenet121',
        #'se_resnet50'
    ]
}

# ======================== 2. 日志配置（每个模型独立日志，便于对比） ========================
def setup_logger(model_name):
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    log_dir = os.path.join(CONFIG['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{model_name}_multiclass_train_{timestamp}.log')
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出
    
    # 文件+控制台双输出
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file

# ======================== 3. 从CSV第三列提取所有类别（保留你的原逻辑） ========================
def extract_all_classes_from_col3(csv_path, logger):
    df = pd.read_csv(csv_path)
    disease_cols = df.columns[2:].tolist()  # 第三列开始的所有疾病列
    CONFIG['all_class_names'] = ['Normal'] + disease_cols
    logger.info(f"✅ 提取所有类别（含无样本）：共{len(CONFIG['all_class_names'])}类")
    logger.info(f"类别列表：{CONFIG['all_class_names']}")
    return df

# ======================== 4. 自定义多分类数据集（保留你的原逻辑） ========================
class RFMiDMulticlassDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.all_class_names = CONFIG['all_class_names']
        self.disease_cols = self.all_class_names[1:]  # 排除Normal的疾病列
        
        # 多分类标签生成逻辑（完全保留你的修复后逻辑）
        self.img_ids = self.csv_df['ID'].values
        self.labels = []
        for _, row in self.csv_df.iterrows():
            if row['Disease_Risk'] == 0:
                self.labels.append(0)  # Normal类（标签0）
            else:
                disease_label = -1
                for cls_idx, cls_name in enumerate(self.disease_cols, 1):
                    if row[cls_name] == 1:
                        disease_label = cls_idx
                        break
                # 无匹配疾病列时标记为最后一类
                self.labels.append(disease_label if disease_label != -1 else len(self.all_class_names)-1)
        
        # 图片路径映射
        self.img_path_dict = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith(CONFIG['img_ext'])
        }

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.img_path_dict[img_id]
        
        # 处理损坏图片
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.fromarray(np.random.randint(0, 255, size=CONFIG['img_size'] + (3,), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ======================== 5. 数据预处理/增强（保留你的原逻辑） ========================
def get_multiclass_transforms(train=True):
    transform_list = [
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        np.random.shuffle(augmentations)
        selected_augs = augmentations[:np.random.randint(2, 4)]
        transform_list = selected_augs + transform_list
    
    return transforms.Compose(transform_list)

# ======================== 6. 创建DataLoader（保留你的原逻辑+日志输出） ========================
def create_multiclass_dataloaders(logger):
    train_transform = get_multiclass_transforms(train=True)
    val_test_transform = get_multiclass_transforms(train=False)
    
    # 数据集实例
    train_dataset = RFMiDMulticlassDataset(
        csv_path=CONFIG['train_csv'],
        image_dir=CONFIG['train_img'],
        transform=train_transform
    )
    val_dataset = RFMiDMulticlassDataset(
        csv_path=CONFIG['val_csv'],
        image_dir=CONFIG['val_img'],
        transform=val_test_transform
    )
    test_dataset = RFMiDMulticlassDataset(
        csv_path=CONFIG['test_csv'],
        image_dir=CONFIG['test_img'],
        transform=val_test_transform
    )
    
    # 获取实际有样本的类别（保留你的逻辑）
    all_actual_labels = list(set(
        train_dataset.labels + val_dataset.labels + test_dataset.labels
    ))
    all_actual_labels.sort()
    CONFIG['actual_class_names'] = [CONFIG['all_class_names'][label] for label in all_actual_labels]
    
    logger.info(f"\n✅ 实际有样本的类别：共{len(CONFIG['actual_class_names'])}类")
    logger.info(f"实际类别列表：{CONFIG['actual_class_names']}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size']*2,
        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size']*2,
        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    logger.info(f"\n✅ DataLoader信息：")
    logger.info(f" - 训练集：{len(train_dataset)}样本 | {len(train_loader)}批次")
    logger.info(f" - 验证集：{len(val_dataset)}样本 | {len(val_loader)}批次")
    logger.info(f" - 测试集：{len(test_dataset)}样本 | {len(test_loader)}批次")
    
    return train_loader, val_loader, test_loader

# ======================== 7. 多分类模型创建（核心新增：7个模型适配多分类） ========================
## 辅助模块：SE模块（用于SE-Net 2017）
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def create_multiclass_model(model_name, num_classes, logger):
    """创建多分类模型（输出维度=所有类别数）"""
    logger.info(f"🔧 初始化多分类模型：{model_name}（输出类别数：{num_classes}）")
    
    if model_name == 'lenet5':
        # CNN 1998（LeNet-5）：多分类适配
        class LeNet5(nn.Module):
            def __init__(self, num_classes):
                super(LeNet5, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5, stride=1),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 120, kernel_size=5, stride=1),
                    nn.Tanh()
                )
                # 适配224×224输入（2次池化后：224→112→56）
                self.classifier = nn.Sequential(
                    nn.Linear(120 * 56 * 56, 84),
                    nn.Tanh(),
                    nn.Linear(84, num_classes)  # 多分类输出
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = LeNet5(num_classes=num_classes)
        logger.info("✅ 模型细节：LeNet-5（1998 CNN）| 无预训练 | 卷积×3+池化×2")
    
    elif model_name == 'alexnet':
        # AlexNet 2012：多分类适配
        model = models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # 1000类→多分类
        logger.info("✅ 模型细节：AlexNet（2012）| 预训练权重 | ReLU+Dropout")
    
    elif model_name == 'vgg16':
        # VGGNet 2014：多分类适配
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # 1000类→多分类
        logger.info("✅ 模型细节：VGG16（2014）| 预训练权重 | 3×3小卷积堆叠")
    
    elif model_name == 'inception_v3':
        # GoogLeNet/Inception 2014：多分类适配
        model = models.inception_v3(pretrained=True, aux_logits=False, transform_input=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1000类→多分类
        logger.info("✅ 模型细节：InceptionV3（2014）| 预训练权重 | 多尺度特征融合")
        logger.warning("⚠️  建议：InceptionV3最佳输入尺寸299×299，可修改CONFIG['img_size']提升性能")
    
    elif model_name == 'resnet50':
        # ResNet 2015：多分类适配
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 1000类→多分类
        logger.info("✅ 模型细节：ResNet50（2015）| 预训练权重 | 残差连接")
    
    elif model_name == 'densenet121':
        # DenseNet 2017：多分类适配
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)  # 1000类→多分类
        logger.info("✅ 模型细节：DenseNet121（2017）| 预训练权重 | 密集连接+特征复用")
    
    elif model_name == 'se_resnet50':
        # SE-Net 2017：多分类适配
        class SEBottleneck(nn.Module):
            expansion = 4
            def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
                super(SEBottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                self.se = SEBlock(planes * self.expansion, reduction)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride
            
            def forward(self, x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.conv3(out)
                out = self.bn3(out)
                out = self.se(out)
                if self.downsample is not None:
                    residual = self.downsample(x)
                out += residual
                out = self.relu(out)
                return out
        
        # 构建SE-ResNet50（多分类输出）
        from torchvision.models.resnet import ResNet
        model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
        # 加载ResNet50预训练权重
        resnet50_pretrained = models.resnet50(pretrained=True)
        pretrained_state = resnet50_pretrained.state_dict()
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and 'se.' not in k}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        logger.info("✅ 模型细节：SE-ResNet50（2017）| 预训练权重 | 通道注意力+残差连接")
    
    else:
        raise ValueError(f"❌ 不支持的模型：{model_name}（请从CONFIG['models_to_train']中选择）")
    
    return model.to(CONFIG['device'])

# ======================== 8. 多分类评估函数（保留你的原逻辑+日志输出） ========================
def evaluate_multiclass(model, loader, split_name, logger):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # 过滤无样本类别
    actual_labels = list(set(all_targets + all_preds))
    actual_labels.sort()
    actual_class_names = [CONFIG['all_class_names'][label] for label in actual_labels]
    
    # 输出分类报告（日志+控制台）
    logger.info(f"\n{split_name} 分类报告（仅显示有样本的类别）：")
    report = classification_report(
        all_targets, all_preds,
        labels=actual_labels,
        target_names=actual_class_names,
        digits=2,
        zero_division=0
    )
    logger.info(report)
    print(report)  # 控制台同步输出
    
    # 返回报告字典（用于保存最佳模型）
    return classification_report(
        all_targets, all_preds,
        labels=actual_labels,
        target_names=actual_class_names,
        output_dict=True,
        zero_division=0
    )

# ======================== 9. 训练函数（保留你的原逻辑+日志输出） ========================
def train_one_epoch(model, loader, criterion, optimizer, epoch, logger):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        # 每10个batch打印进度（日志+控制台）
        if (batch_idx + 1) % 10 == 0:
            log_msg = f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}"
            logger.info(log_msg)
            print(log_msg)
    
    avg_loss = total_loss / len(loader.dataset)
    log_msg = f"Epoch [{epoch+1}] 训练损失：{avg_loss:.4f}"
    logger.info(log_msg)
    print(log_msg)
    return avg_loss

# ======================== 10. 单个模型训练流程（集成日志+模型保存） ========================
def train_single_multiclass_model(model_name):
    # 1. 初始化日志
    logger, log_file = setup_logger(model_name)
    logger.info(f"{'='*60}")
    logger.info(f"开始训练多分类模型：{model_name}")
    logger.info(f"训练配置：{CONFIG}")
    logger.info(f"{'='*60}\n")
    
    try:
        # 2. 提取所有类别
        extract_all_classes_from_col3(CONFIG['train_csv'], logger)
        
        # 3. 创建DataLoader
        train_loader, val_loader, test_loader = create_multiclass_dataloaders(logger)
        
        # 4. 初始化模型（多分类输出维度=所有类别数）
        num_classes = len(CONFIG['all_class_names'])
        model = create_multiclass_model(model_name, num_classes, logger)
        
        # 5. 初始化损失函数、优化器、调度器
        criterion = nn.CrossEntropyLoss()  # 多分类标准损失
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # 6. 训练记录（保存最佳模型）
        best_val_macro_f1 = 0.0
        model_save_dir = os.path.join(CONFIG['output_dir'], 'best_models')
        os.makedirs(model_save_dir, exist_ok=True)
        best_model_path = os.path.join(model_save_dir, f'{model_name}_multiclass_best.pth')
        
        # 7. 训练循环
        for epoch in range(CONFIG['epochs']):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{CONFIG['epochs']} {'='*40}")
            print(f"\n{'='*40} Epoch {epoch+1}/{CONFIG['epochs']} {'='*40}")
            
            # 训练
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger)
            
            # 验证
            val_report = evaluate_multiclass(model, val_loader, "Validation", logger)
            
            # 基于宏平均F1保存最佳模型（避免类别不平衡影响）
            val_macro_f1 = val_report['macro avg']['f1-score']
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                torch.save({
                    'model_name': model_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_macro_f1': best_val_macro_f1,
                    'all_class_names': CONFIG['all_class_names'],
                    'actual_class_names': CONFIG['actual_class_names'],
                    'config': CONFIG
                }, best_model_path)
                logger.info(f"✅ 保存最佳模型（验证宏平均F1: {best_val_macro_f1:.4f}）到：{best_model_path}")
                print(f"✅ 保存最佳模型（验证宏平均F1: {best_val_macro_f1:.4f}）")
            
            # 学习率调度
            scheduler.step(train_loss)
        
        # 8. 测试集最终评估（加载最佳模型）
        logger.info(f"\n{'='*60}")
        logger.info(f"测试集最终评估（加载最佳模型）")
        logger.info(f"{'='*60}")
        print(f"\n{'='*60}")
        print(f"测试集最终评估（加载最佳模型）")
        print(f"{'='*60}")
        
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_multiclass(model, test_loader, "Test", logger)
        
        # 9. 训练总结
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name} 训练总结：")
        logger.info(f" - 最佳验证宏平均F1：{best_val_macro_f1:.4f}")
        logger.info(f" - 最佳模型路径：{best_model_path}")
        logger.info(f" - 日志文件路径：{log_file}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ 训练过程出错：{str(e)}", exc_info=True)
        raise

# ======================== 11. 多模型批量训练（主流程） ========================
def main():
    print(f"{'='*70}")
    print(f"RFMiD多分类多模型比对训练")
    print(f"训练模型列表：{CONFIG['models_to_train']}")
    print(f"设备：{CONFIG['device']} | 输出目录：{CONFIG['output_dir']}")
    print(f"{'='*70}\n")
    
    # 循环训练每个模型
    for model_name in CONFIG['models_to_train']:
        print(f"\n{'='*80}")
        print(f"正在训练多分类模型：{model_name}")
        print(f"{'='*80}")
        
        train_single_multiclass_model(model_name)
        
        print(f"\n✅ {model_name} 训练完成！结果保存到：{CONFIG['output_dir']}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*70}")
    print(f"所有多分类模型训练完成！")
    print(f"结果汇总：")
    print(f" - 日志文件：{os.path.join(CONFIG['output_dir'], 'logs')}")
    print(f" - 最佳模型：{os.path.join(CONFIG['output_dir'], 'best_models')}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()