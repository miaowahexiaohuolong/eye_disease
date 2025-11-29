import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime
import efficientnet_pytorch  # 需额外安装：pip install efficientnet-pytorch

# ======================== 1. 核心配置（修改这里适配你的环境） ========================
CONFIG = {
    # 路径配置
    'train_csv': r'E:\eyes__disease\dataset\Training_Set\RFMiD_Training_Labels.csv',
    'train_img': r'E:\eyes__disease\dataset\Training_Set\Training',
    'val_csv': r'E:\eyes__disease\dataset\Evaluation_Set\RFMiD_Validation_Labels.csv',
    'val_img': r'E:\eyes__disease\dataset\Evaluation_Set\Validation',
    'test_csv': r'E:\eyes__disease\dataset\Test_Set\RFMiD_Testing_Labels.csv',
    'test_img': r'E:\eyes__disease\dataset\Test_Set\Test',
    
    # 训练参数
    'img_size': (224, 224),    # 图片尺寸（InceptionV3建议改为299×299，其他模型兼容224）
    'batch_size': 16,          # 适配多模型（显存不足可改8；VGG16/InceptionV3建议8）
    'num_workers': 12,          # CPU核心数
    'img_ext': '.png',          # 图片扩展名
    'epochs': 30,              # 训练轮次
    'lr': 1e-4,                # 学习率（InceptionV3/SE-ResNet50建议5e-5）
    'weight_decay': 1e-5,       # 权重衰减（防过拟合）
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 自动检测GPU/CPU
    'output_dir': 'model_comparison_results',  # 日志和模型保存根目录
    'models_to_train': [        # 要比对的经典模型列表（可增删）

        #'resnet50', # ResNet 2015
        #'densenet121', # DenseNet 2017
        #'vgg16'
        'alexnet'
    ]
}

# ======================== 2. 日志配置（为每个模型创建独立日志） ========================
def setup_logger(model_name):
    """创建模型专属日志器，保存训练记录"""
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    log_dir = os.path.join(CONFIG['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名：模型名_时间.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{model_name}_train_{timestamp}.log')
    
    # 配置日志格式
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出
    
    # 文件处理器（保存到日志文件）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 控制台处理器（同时输出到控制台）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ======================== 3. 自定义二分类数据集（保持原逻辑） ========================
class RFMiDBinaryDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # 二分类标签：Disease_Risk（0=正常，1=疾病）
        self.labels = self.csv_df['Disease_Risk'].values
        self.img_ids = self.csv_df['ID'].values
        
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
        
        # 读取图片（转为RGB，处理损坏图片）
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.fromarray(np.random.randint(0, 255, size=CONFIG['img_size'] + (3,), dtype=np.uint8))
        
        # 应用预处理/增强
        if self.transform:
            image = self.transform(image)
        
        # 二分类标签（float32适配BCELoss）
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

# ======================== 4. 数据预处理/增强（保持原逻辑） ========================
def get_binary_transforms(train=True):
    transform_list = [
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet归一化
                           std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        # 随机添加增强（每次训练随机触发）
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        np.random.shuffle(augmentations)
        selected_augs = augmentations[:np.random.randint(1, 3)]
        transform_list = selected_augs + transform_list
    
    return transforms.Compose(transform_list)

# ======================== 5. 创建DataLoader（保持原逻辑） ========================
def create_binary_dataloaders(logger):
    train_transform = get_binary_transforms(train=True)
    val_test_transform = get_binary_transforms(train=False)
    
    # 数据集实例
    train_dataset = RFMiDBinaryDataset(
        csv_path=CONFIG['train_csv'],
        image_dir=CONFIG['train_img'],
        transform=train_transform
    )
    val_dataset = RFMiDBinaryDataset(
        csv_path=CONFIG['val_csv'],
        image_dir=CONFIG['val_img'],
        transform=val_test_transform
    )
    test_dataset = RFMiDBinaryDataset(
        csv_path=CONFIG['test_csv'],
        image_dir=CONFIG['test_img'],
        transform=val_test_transform
    )
    
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
    
    logger.info(f"✅ DataLoader创建完成：")
    logger.info(f" - 训练集：{len(train_dataset)}样本 | {len(train_loader)}批次")
    logger.info(f" - 验证集：{len(val_dataset)}样本 | {len(val_loader)}批次")
    logger.info(f" - 测试集：{len(test_dataset)}样本 | {len(test_loader)}批次")
    
    return train_loader, val_loader, test_loader

# ======================== 6. 多个经典模型创建函数（核心新增所有目标模型） ========================
## 辅助模块：SE模块（用于SE-Net 2017）
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 挤压：全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()  # 激励：输出注意力权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # (b,c,h,w) → (b,c)
        y = self.fc(y).view(b, c, 1, 1)  # (b,c) → (b,c,1,1)
        return x * y.expand_as(x)  # 注意力权重乘原特征图

def create_model(model_name, logger):
    """根据模型名创建对应的二分类模型（新增所有目标模型）"""
    logger.info(f"🔧 初始化模型：{model_name}")
    num_classes = 1  # 二分类输出
    
    if model_name == 'lenet5':
        # CNN 1998（LeNet-5）：最早CNN，奠定基础
        class LeNet5(nn.Module):
            def __init__(self, num_classes=1):
                super(LeNet5, self).__init__()
                # 特征提取：卷积+池化（经典LeNet结构）
                self.features = nn.Sequential(
                    nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),  # RGB输入→6通道
                    nn.Tanh(),  # 原始用sigmoid，Tanh更稳定
                    nn.AvgPool2d(kernel_size=2, stride=2),  # 尺寸/2
                    nn.Conv2d(6, 16, kernel_size=5, stride=1),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=2),  # 尺寸/2
                    nn.Conv2d(16, 120, kernel_size=5, stride=1),
                    nn.Tanh()
                )
                # 分类层：适配224×224输入（2次池化后尺寸：224→112→56）
                self.classifier = nn.Sequential(
                    nn.Linear(120 * 56 * 56, 84),  # 120×56×56 = 376320
                    nn.Tanh(),
                    nn.Linear(84, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)  # 展平
                x = self.classifier(x)
                return x
        
        model = LeNet5(num_classes=num_classes)
        logger.info("✅ 模型细节：LeNet-5（1998经典CNN）| 无预训练 | 卷积×3+池化×2+全连接×2")
    
    elif model_name == 'alexnet':
        # AlexNet 2012：ReLU、Dropout、GPU训练
        model = models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # 1000类→1类
        logger.info("✅ 模型细节：AlexNet（2012）| 预训练权重 | ReLU+Dropout+GPU适配")
    
    elif model_name == 'vgg16':
        # VGGNet 2014：3×3小卷积堆叠，结构简洁
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)  # 1000类→1类
        logger.info("✅ 模型细节：VGG16（2014）| 预训练权重 | 3×3小卷积堆叠+全连接层")
    
    elif model_name == 'inception_v3':
        # GoogLeNet/Inception 2014：Inception模块，多尺度特征
        # 注意：InceptionV3默认输入尺寸≥299，若用224需设置transform_input=True
        model = models.inception_v3(pretrained=True, aux_logits=False, transform_input=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1000类→1类
        logger.info("✅ 模型细节：InceptionV3（2014）| 预训练权重 | Inception模块+多尺度特征融合")
        logger.warning("⚠️  建议：InceptionV3最佳输入尺寸299×299，可修改CONFIG['img_size']提升性能")
    
    elif model_name == 'resnet50':
        # ResNet 2015：残差连接，解决梯度消失
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 1000类→1类
        logger.info("✅ 模型细节：ResNet50（2015）| 预训练权重 | 残差连接+深层网络")
    
    elif model_name == 'densenet121':
        # DenseNet 2017：密集连接，特征复用
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)  # 1000类→1类
        logger.info("✅ 模型细节：DenseNet121（2017）| 预训练权重 | 密集连接+特征复用")
    
    elif model_name == 'se_resnet50':
        # SE-Net 2017：通道注意力机制，即插即用
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
                self.se = SEBlock(planes * self.expansion, reduction)  # 插入SE模块
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
                out = self.se(out)  # 应用通道注意力
                if self.downsample is not None:
                    residual = self.downsample(x)
                out += residual  # 残差连接
                out = self.relu(out)
                return out
        
        # 构建SE-ResNet50
        from torchvision.models.resnet import ResNet
        model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
        # 加载ResNet50预训练权重（过滤SE模块的权重）
        resnet50_pretrained = models.resnet50(pretrained=True)
        pretrained_state = resnet50_pretrained.state_dict()
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and 'se.' not in k}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        logger.info("✅ 模型细节：SE-ResNet50（2017）| 预训练权重 | 通道注意力+残差连接")
    
    else:
        raise ValueError(f"❌ 不支持的模型：{model_name}（请从CONFIG['models_to_train']中选择）")
    
    # 移到设备
    model = model.to(CONFIG['device'])
    logger.info(f"✅ 模型初始化完成，设备：{CONFIG['device']}")
    return model

# ======================== 7. 评价指标计算（保持原逻辑） ========================
def calculate_metrics(preds, targets):
    preds_binary = (preds > 0.5).float().cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    precision = precision_score(targets_np, preds_binary, zero_division=0)
    recall = recall_score(targets_np, preds_binary, zero_division=0)
    f1 = f1_score(targets_np, preds_binary, zero_division=0)
    
    return {
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }

# ======================== 8. 训练/验证/测试函数（集成日志） ========================
def train_one_epoch(model, loader, criterion, optimizer, epoch, logger):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach())
        all_targets.extend(labels.detach())
        
        # 每10个batch打印日志
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(torch.stack(all_preds), torch.stack(all_targets))
    
    logger.info(f"Epoch [{epoch+1}] 训练结果 | Loss: {avg_loss:.4f} | Precision: {metrics['Precision']} | Recall: {metrics['Recall']} | F1: {metrics['F1']}")
    return avg_loss, metrics

def evaluate(model, loader, criterion, split_name, logger):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(torch.sigmoid(outputs).detach())
            all_targets.extend(labels.detach())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(torch.stack(all_preds), torch.stack(all_targets))
    
    logger.info(f"\n{split_name} 结果 | Loss: {avg_loss:.4f} | Precision: {metrics['Precision']} | Recall: {metrics['Recall']} | F1: {metrics['F1']}\n")
    return avg_loss, metrics

# ======================== 9. 单个模型训练流程（集成日志和模型保存） ========================
def train_single_model(model_name):
    # 1. 初始化日志
    logger, log_file = setup_logger(model_name)
    logger.info(f"{'='*60}")
    logger.info(f"开始训练模型：{model_name}")
    logger.info(f"训练配置：{CONFIG}")
    logger.info(f"{'='*60}\n")
    
    try:
        # 2. 创建DataLoader
        train_loader, val_loader, test_loader = create_binary_dataloaders(logger)
        
        # 3. 创建模型
        model = create_model(model_name, logger)
        
        # 4. 初始化损失函数、优化器、调度器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # 5. 训练记录
        best_val_f1 = 0.0
        model_save_dir = os.path.join(CONFIG['output_dir'], 'best_models')
        os.makedirs(model_save_dir, exist_ok=True)
        best_model_path = os.path.join(model_save_dir, f'{model_name}_best.pth')
        
        # 6. 训练循环
        for epoch in range(CONFIG['epochs']):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{CONFIG['epochs']} {'='*40}")
            
            # 训练
            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger)
            
            # 验证
            val_loss, val_metrics = evaluate(model, val_loader, criterion, "Validation", logger)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 保存最佳模型（基于验证集F1）
            if val_metrics['F1'] > best_val_f1:
                best_val_f1 = val_metrics['F1']
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'val_metrics': val_metrics,
                    'config': CONFIG
                }, best_model_path)
                logger.info(f"✅ 保存最佳模型（F1: {best_val_f1:.4f}）到：{best_model_path}")
        
        # 7. 测试集最终评估（加载最佳模型）
        logger.info(f"\n{'='*60}")
        logger.info(f"测试集最终评估（加载最佳模型）")
        logger.info(f"{'='*60}")
        
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_metrics = evaluate(model, test_loader, criterion, "Test", logger)
        
        # 8. 训练总结
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name} 训练总结：")
        logger.info(f" - 最佳验证F1：{best_val_f1:.4f}")
        logger.info(f" - 测试集Precision：{test_metrics['Precision']}")
        logger.info(f" - 测试集Recall：{test_metrics['Recall']}")
        logger.info(f" - 测试集F1：{test_metrics['F1']}")
        logger.info(f" - 最佳模型路径：{best_model_path}")
        logger.info(f" - 日志文件路径：{log_file}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ 训练过程出错：{str(e)}", exc_info=True)
        raise

# ======================== 10. 多模型批量训练（主流程） ========================
def main():
    print(f"{'='*70}")
    print(f"开始多经典模型比对训练（二分类：Normal vs Disease）")
    print(f"训练模型列表：{CONFIG['models_to_train']}")
    print(f"设备：{CONFIG['device']} | 输出目录：{CONFIG['output_dir']}")
    print(f"{'='*70}\n")
    
    # 循环训练每个模型
    for model_name in CONFIG['models_to_train']:
        print(f"\n{'='*80}")
        print(f"正在训练模型：{model_name}")
        print(f"{'='*80}")
        
        train_single_model(model_name)
        
        print(f"\n✅ {model_name} 训练完成！日志和模型已保存到：{CONFIG['output_dir']}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*70}")
    print(f"所有模型训练完成！")
    print(f"结果汇总：")
    print(f" - 日志文件：{os.path.join(CONFIG['output_dir'], 'logs')}")
    print(f" - 最佳模型：{os.path.join(CONFIG['output_dir'], 'best_models')}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()