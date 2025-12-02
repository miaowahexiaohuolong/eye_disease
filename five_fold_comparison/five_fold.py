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
from sklearn.model_selection import KFold  # 新增：K折分割
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime
import csv

# ======================== 1. 核心配置（新增K折参数） ========================
CONFIG = {
    # 路径配置（保持你的原始路径）
    'train_csv': '/root/autodl-tmp/dataset/Training_Set/RFMiD_Training_Labels.csv',
    'train_img': '/root/autodl-tmp/dataset/Training_Set/Training',
    'val_csv': '/root/autodl-tmp/dataset/Evaluation_Set/RFMiD_Validation_Labels.csv',
    'val_img': '/root/autodl-tmp/dataset/Evaluation_Set/Validation',
    'test_csv': '/root/autodl-tmp/dataset/Test_Set/RFMiD_Testing_Labels.csv',
    'test_img': '/root/autodl-tmp/dataset/Test_Set/Test',
    
    # 多分类相关配置
    'all_class_names': [],
    'actual_class_names': [],
    'img_ext': '.png',          
    'img_size': (224, 224),
    'batch_size': 32,
    'num_workers': 16,          
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 1e-5,       
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'multiclass_kfold_results',  # 修改目录名
    'models_to_train': ['resnet50'],  # 可添加更多模型
    'n_splits': 5,  # 新增：5折交叉验证
}


# ======================== 2. 日志配置（支持折数标识） ========================
def setup_logger(model_name, fold_idx=None):
    """
    创建模型专属日志器
    fold_idx: 折数索引（None表示汇总日志）
    """
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    log_dir = os.path.join(CONFIG['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if fold_idx is not None:
        log_file = os.path.join(log_dir, f'{model_name}_fold{fold_idx+1}_train_{timestamp}.log')
        logger_name = f'{model_name}_fold{fold_idx+1}'
    else:
        log_file = os.path.join(log_dir, f'{model_name}_summary_{timestamp}.log')
        logger_name = f'{model_name}_summary'
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
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

# ======================== 3. 提取所有类别 ========================
def extract_all_classes_from_col3(csv_path, logger):
    df = pd.read_csv(csv_path)
    disease_cols = df.columns[2:].tolist()
    CONFIG['all_class_names'] = ['Normal'] + disease_cols
    logger.info(f"✅ 提取所有类别（含无样本）：共{len(CONFIG['all_class_names'])}类")
    return df

# ======================== 4. 自定义多分类数据集（增强版） ========================
class RFMiDMulticlassDataset(Dataset):
    def __init__(self, csv_path, image_dir=None, transform=None, subset_idx=None):
        ### 修复：支持合并CSV路径和子集索引
        self.csv_df = pd.read_csv(csv_path)
        self.transform = transform
        self.all_class_names = CONFIG['all_class_names']
        self.disease_cols = self.all_class_names[1:]
        
        ### 修复：支持K折子集
        if subset_idx is not None:
            self.csv_df = self.csv_df.iloc[subset_idx].reset_index(drop=True)
        
        # 标签生成逻辑（保持原逻辑）
        self.img_ids = self.csv_df['ID'].values
        self.labels = []
        for _, row in self.csv_df.iterrows():
            if row['Disease_Risk'] == 0:
                self.labels.append(0)
            else:
                disease_label = -1
                for cls_idx, cls_name in enumerate(self.disease_cols, 1):
                    if row[cls_name] == 1:
                        disease_label = cls_idx
                        break
                self.labels.append(disease_label if disease_label != -1 else len(self.all_class_names)-1)
        
        # 图片路径映射（优化：支持多目录扫描）
        self.img_path_dict = {}
        img_dirs_to_scan = []
        if image_dir is not None:
            if isinstance(image_dir, (list, tuple)):
                img_dirs_to_scan = list(image_dir)
            else:
                img_dirs_to_scan = [image_dir]
        else:
            img_dirs_to_scan = [CONFIG['train_img'], CONFIG['val_img']]
        
        for img_dir in img_dirs_to_scan:
            if os.path.exists(img_dir):
                self.img_path_dict.update({
                    int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
                    for f in os.listdir(img_dir) if f.endswith(CONFIG['img_ext'])
                })

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.img_path_dict[img_id]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.fromarray(np.random.randint(0, 255, size=CONFIG['img_size'] + (3,), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ======================== 5. 数据预处理/增强 ========================
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

# ======================== 6. 创建K折DataLoader（核心新增） ========================
def create_kfold_dataloaders(logger, n_splits=5):
    """
    创建K折交叉验证的数据加载器
    返回：kfold_splitter, combined_csv_path, test_loader
    """
    # 合并训练集和验证集CSV
    train_df = pd.read_csv(CONFIG['train_csv'])
    val_df = pd.read_csv(CONFIG['val_csv'])
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # 保存合并后的数据并返回路径
    combined_csv_path = os.path.join(CONFIG['output_dir'], 'combined_train_val_multiclass.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info(f"合并训练+验证集：{len(train_df)} + {len(val_df)} = {len(combined_df)} 样本")
    logger.info(f"合并CSV已保存至：{combined_csv_path}")
    
    # 创建K折分割器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_indices = list(kf.split(combined_df))
    
    # 测试集保持不变
    test_transform = get_multiclass_transforms(train=False)
    test_dataset = RFMiDMulticlassDataset(
        csv_path=CONFIG['test_csv'],
        image_dir=CONFIG['test_img'],
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size']*2,
        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    logger.info(f"✅ K折分割完成：{n_splits}折 | 每折训练/验证约 {int(len(combined_df)*0.8)}/{int(len(combined_df)*0.2)} 样本")
    logger.info(f"✅ 测试集加载：{len(test_dataset)} 样本")
    
    return split_indices, combined_csv_path, test_loader

# ======================== 7. 多分类模型创建 ========================
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
    """创建多分类模型"""
    logger.info(f"🔧 初始化多分类模型：{model_name}（输出类别数：{num_classes}）")
    
    if model_name == 'lenet5':
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
                self.classifier = nn.Sequential(
                    nn.Linear(120 * 56 * 56, 84),
                    nn.Tanh(),
                    nn.Linear(84, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = LeNet5(num_classes=num_classes)
        logger.info("✅ 模型细节：LeNet-5（1998 CNN）| 无预训练")
    
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        logger.info("✅ 模型细节：AlexNet（2012）| 预训练权重")
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        logger.info("✅ 模型细节：VGG16（2014）| 预训练权重")
    
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False, transform_input=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info("✅ 模型细节：InceptionV3（2014）| 预训练权重")
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        logger.info("✅ 模型细节：ResNet50（2015）| 预训练权重")
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        logger.info("✅ 模型细节：DenseNet121（2017）| 预训练权重")
    
    elif model_name == 'se_resnet50':
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
        
        from torchvision.models.resnet import ResNet
        model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
        resnet50_pretrained = models.resnet50(pretrained=True)
        pretrained_state = resnet50_pretrained.state_dict()
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and 'se.' not in k}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        logger.info("✅ 模型细节：SE-ResNet50（2017）| 预训练权重")
    
    else:
        raise ValueError(f"❌ 不支持的模型：{model_name}")
    
    return model.to(CONFIG['device'])

# ======================== 8. 多分类评估函数 ========================
def evaluate_multiclass(model, loader, split_name, logger, criterion=None):
    """评估模型性能：计算损失+生成分类报告"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # 过滤无样本类别
    actual_labels = list(set(all_targets + all_preds))
    actual_labels.sort()
    CONFIG['actual_class_names'] = [CONFIG['all_class_names'][label] for label in actual_labels]
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    logger.info(f"\n{split_name} 分类报告（仅显示有样本的类别）| 损失: {avg_loss:.4f}")
    report = classification_report(
        all_targets, all_preds,
        labels=actual_labels,
        target_names=CONFIG['actual_class_names'],
        digits=2,
        zero_division=0
    )
    logger.info(report)
    print(report)
    
    report_dict = classification_report(
        all_targets, all_preds,
        labels=actual_labels,
        target_names=CONFIG['actual_class_names'],
        output_dict=True,
        zero_division=0
    )
    report_dict['loss'] = avg_loss
    return report_dict

# ======================== 9. 训练函数 ========================
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
        
        if (batch_idx + 1) % 10 == 0:
            log_msg = f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}"
            logger.info(log_msg)
            print(log_msg)
    
    avg_loss = total_loss / len(loader.dataset)
    log_msg = f"Epoch [{epoch+1}] 训练损失：{avg_loss:.4f}"
    logger.info(log_msg)
    print(log_msg)
    return avg_loss

# ======================== 10. K折训练流程（核心改造） ========================
def train_kfold_multiclass_model(model_name):
    """单个模型的5折交叉验证训练流程"""
    print(f"\n{'='*80}")
    print(f"开始训练多分类模型：{model_name}（{CONFIG['n_splits']}折交叉验证）")
    print(f"{'='*80}")
    
    # 1. 创建汇总日志（跨折总结）
    summary_logger, summary_log_file = setup_logger(model_name, fold_idx=None)
    summary_logger.info(f"{'='*70}")
    summary_logger.info(f"模型 {model_name} 开始{CONFIG['n_splits']}折交叉验证")
    summary_logger.info(f"{'='*70}")
    
    # 2. 提取所有类别
    extract_all_classes_from_col3(CONFIG['train_csv'], summary_logger)
    
    # 3. 创建K折分割器和测试集
    split_indices, combined_csv_path, test_loader = create_kfold_dataloaders(summary_logger, CONFIG['n_splits'])
    
    # 4. 跨折结果记录
    fold_results = []
    best_models = []
    
    # 5. 循环每一折
    for fold_idx, (train_idx, val_idx) in enumerate(split_indices):
        print(f"\n{'='*60}")
        print(f"模型 {model_name} | 第 {fold_idx+1}/{CONFIG['n_splits']} 折训练")
        print(f"{'='*60}")
        
        # 为每折创建独立日志
        logger, log_file = setup_logger(model_name, fold_idx)
        logger.info(f"{'='*60}")
        logger.info(f"模型 {model_name} | 第 {fold_idx+1}/{CONFIG['n_splits']} 折")
        logger.info(f"训练样本数：{len(train_idx)} | 验证样本数：{len(val_idx)}")
        logger.info(f"{'='*60}\n")
        
        try:
            # 6. 创建该折的数据加载器
            train_transform = get_multiclass_transforms(train=True)
            val_transform = get_multiclass_transforms(train=False)
            
            train_dataset = RFMiDMulticlassDataset(
                csv_path=combined_csv_path,
                image_dir=None,  # 自动扫描训练和验证目录
                transform=train_transform,
                subset_idx=train_idx
            )
            val_dataset = RFMiDMulticlassDataset(
                csv_path=combined_csv_path,
                image_dir=None,
                transform=val_transform,
                subset_idx=val_idx
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=CONFIG['batch_size'],
                shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=CONFIG['batch_size']*2,
                shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
            )
            
            logger.info(f"✅ 数据加载器创建完成：训练集 {len(train_dataset)} | 验证集 {len(val_dataset)}")
            
            # 7. 初始化该折模型（每折重新初始化）
            num_classes = len(CONFIG['all_class_names'])
            model = create_multiclass_model(model_name, num_classes, logger)
            
            # 8. 初始化损失函数、优化器、调度器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG['lr'],
                weight_decay=CONFIG['weight_decay']
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
            
            # 9. 每折记录CSV（新增）
            records_dir = os.path.join(CONFIG['output_dir'], 'training_records', model_name)
            os.makedirs(records_dir, exist_ok=True)
            fold_record_csv = os.path.join(records_dir, f'{model_name}_fold{fold_idx+1}_losses.csv')
            with open(fold_record_csv, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Macro_F1'])
            
            # 10. 训练记录
            best_val_macro_f1 = 0.0
            model_save_dir = os.path.join(CONFIG['output_dir'], 'best_models', model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_path = os.path.join(model_save_dir, f'{model_name}_fold{fold_idx+1}_best.pth')
            
            # 11. 训练循环
            for epoch in range(CONFIG['epochs']):
                logger.info(f"\n{'='*40} Epoch {epoch+1}/{CONFIG['epochs']} {'='*40}")
                print(f"\n{'='*40} Epoch {epoch+1}/{CONFIG['epochs']} {'='*40}")
                
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger)
                val_report = evaluate_multiclass(model, val_loader, "Validation", logger, criterion)
                val_loss = val_report['loss']
                val_macro_f1 = val_report['macro avg']['f1-score']
                
                # 记录到CSV
                with open(fold_record_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_macro_f1:.4f}"])
                
                scheduler.step(val_loss)
                
                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    torch.save({
                        'fold': fold_idx+1,
                        'epoch': epoch,
                        'model_name': model_name,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_macro_f1': best_val_macro_f1,
                        'val_metrics': val_report,
                        'config': CONFIG
                    }, best_model_path)
                    logger.info(f"✅ 保存最佳模型（F1: { best_val_macro_f1:.4f}）到：{best_model_path}")
                    print(f"✅ 保存最佳模型（F1: {best_val_macro_f1:.4f}）到：{best_model_path}")
            
            # 12. 记录该折结果
            fold_results.append({
                'fold': fold_idx+1,
                'best_val_macro_f1': best_val_macro_f1,
                'val_report': val_report,
                'model_path': best_model_path
            })
            best_models.append(best_model_path)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"第{fold_idx+1}折训练完成！最佳验证宏平均F1：{best_val_macro_f1:.4f}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"❌ 第{fold_idx+1}折训练失败：{str(e)}", exc_info=True)
            continue
    
    # 13. 计算交叉验证平均结果
    if len(fold_results) > 0:
        avg_val_macro_f1 = np.mean([r['best_val_macro_f1'] for r in fold_results])
        std_val_macro_f1 = np.std([r['best_val_macro_f1'] for r in fold_results])
    else:
        avg_val_macro_f1 = std_val_macro_f1 = float('nan')
    
    # 14. 保存交叉验证结果到汇总日志
    summary_logger.info(f"\n{'='*70}")
    summary_logger.info(f"模型 {model_name} {CONFIG['n_splits']}折交叉验证完成")
    summary_logger.info(f"{'='*70}")
    for result in fold_results:
        summary_logger.info(f"第{result['fold']}折 | 验证宏平均F1: {result['best_val_macro_f1']:.4f} | 模型: {result['model_path']}")
    summary_logger.info(f"\n{'='*70}")
    summary_logger.info(f"平均验证宏平均F1：{avg_val_macro_f1:.4f} ± {std_val_macro_f1:.4f}")
    summary_logger.info(f"{'='*70}")
    
    print(f"\n✅ {model_name} 5折交叉验证完成！平均验证宏平均F1：{avg_val_macro_f1:.4f} ± {std_val_macro_f1:.4f}")
    
    # 15. 测试集评估（使用第1折模型）
    if len(best_models) > 0:
        print(f"\n{'='*70}")
        print(f"使用第1折最佳模型进行测试集评估")
        print(f"{'='*70}")
        
        logger, _ = setup_logger(model_name, 0)
        checkpoint = torch.load(best_models[0])
        model = create_multiclass_model(model_name, num_classes, logger)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        criterion = nn.CrossEntropyLoss()
        final_test_report = evaluate_multiclass(model, test_loader, "Final Test", logger, criterion)
        
        summary_logger.info(f"\n{'='*70}")
        summary_logger.info(f"测试集最终评估结果（使用第1折模型）")
        summary_logger.info(f"损失: {final_test_report['loss']:.4f} | 宏平均F1: {final_test_report['macro avg']['f1-score']:.4f}")
        summary_logger.info(f"{'='*70}")
        
        print(f"测试集结果：损失={final_test_report['loss']:.4f} 宏平均F1={final_test_report['macro avg']['f1-score']:.4f}")
    
    return fold_results, summary_log_file

# ======================== 11. 多模型批量训练（主流程） ========================
def main():
    print(f"{'='*70}")
    print(f"RFMiD多分类多模型5折交叉验证训练")
    print(f"训练模型列表：{CONFIG['models_to_train']}")
    print(f"设备：{CONFIG['device']} | 输出目录：{CONFIG['output_dir']}")
    print(f"{'='*70}\n")
    
    all_models_results = {}
    
    for model_name in CONFIG['models_to_train']:
        print(f"\n{'='*80}")
        print(f"正在训练多分类模型：{model_name}")
        print(f"{'='*80}")
        
        fold_results, summary_log = train_kfold_multiclass_model(model_name)
        all_models_results[model_name] = {
            'fold_results': fold_results,
            'summary_log': summary_log
        }
        
        print(f"\n✅ {model_name} 训练完成！日志已保存到：{summary_log}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*70}")
    print(f"所有模型5折交叉验证完成！")
    print(f"结果汇总：")
    print(f" - 详细日志：{os.path.join(CONFIG['output_dir'], 'logs')}")
    print(f" - 最佳模型：{os.path.join(CONFIG['output_dir'], 'best_models')}")
    print(f" - 损失记录：{os.path.join(CONFIG['output_dir'], 'training_records')}")
    print(f" - 各模型平均性能：")
    for model_name, results in all_models_results.items():
        fold_results = results['fold_results']
        if len(fold_results) > 0:
            avg_f1 = np.mean([r['best_val_macro_f1'] for r in fold_results])
            std_f1 = np.std([r['best_val_macro_f1'] for r in fold_results])
            print(f"   * {model_name}: {avg_f1:.4f} ± {std_f1:.4f}")
        else:
            print(f"   * {model_name}: 训练失败")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()