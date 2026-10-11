# -*- coding: utf-8 -*-
"""
改进的 6 类分类 CNN 

新增指标：
1. Sensitivity (Recall) for each class
2. Specificity for each class  
3. Precision for each class
4. F1-score for each class
5. Overall Accuracy (OA)
6. 完整的混淆矩阵元素分析

@author: wenjie (improved)
"""

from __future__ import print_function
import time
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# ------------------------------ 参数设置 ------------------------------
seeds = range(1, 11)
results = []

for seed in seeds:
    print(f"当前种子: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 256  
    epochs = 10  # 6分类使用更多训练轮数
    img_size = 20
    da = 100  
    num_classes = 6  # 6分类
    
    # 超参数：融合CL4和CL6的优势
    label_smoothing = 0.1  
    dropout_rate = 0.5     
    weight_decay = 0.05   

    # ---------------------- 数据加载与预处理 ------------------------------
    data_file = 'sda_dat/uniref100/CL317_uniref100_pssm20_r0Des_60mer_25gapcbd_DA150_DL00.npz'
    alldata = np.load(data_file)
    datX = alldata['datX']
    ylab = alldata['ylab']

    datX = datX[:, :da, :, :]
    ylab = ylab[:, :da]
    box = datX.shape[1]  

    def convert_data_dl(train_X, test_X, train_y, test_y):
        n_train = train_X.shape[0]
        train_X = train_X.reshape(n_train * box, 1, img_size, img_size).astype(np.float32)
        train_y = train_y.reshape(n_train * box).astype(np.int64)
        train_y = train_y - 1  # 转为0-based
        
        n_test = test_X.shape[0]
        test_X = test_X.reshape(n_test * box, 1, img_size, img_size).astype(np.float32)
        test_y = test_y.reshape(n_test * box).astype(np.int64)
        test_y = test_y - 1
        
        train_dataset = CNN_Dataset(train_X, train_y)
        test_dataset = CNN_Dataset(test_X, test_y)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
        
        return train_loader, test_loader, box, test_y

    class CNN_Dataset(Dataset):
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y)
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        def __len__(self):
            return self.x_data.shape[0]

    # --------------------------- 改进的网络架构 ------------------------------
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            residual = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            return out

    class ImprovedNet(nn.Module):
        def __init__(self, num_classes=6, dropout_rate=0.5):
            super().__init__()
            self.conv_init = nn.Conv2d(1, 32, 3, padding=1)
            self.bn_init = nn.BatchNorm2d(32)
            
            self.res_block1 = ResidualBlock(32, 32)
            self.res_block2 = ResidualBlock(32, 64)
            self.res_block3 = ResidualBlock(64, 64)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(64, 32)
            self.bn_fc = nn.BatchNorm1d(32)
            self.fc2 = nn.Linear(32, num_classes)
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
        
        def forward(self, x):
            x = F.relu(self.bn_init(self.conv_init(x)))
            x = self.res_block1(x)
            x = F.max_pool2d(x, 2)
            x = self.res_block2(x)
            x = F.max_pool2d(x, 2)
            x = self.res_block3(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = F.relu(self.bn_fc(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # ------------------------- Focal Loss + Label Smoothing ------------------------------
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2, label_smoothing=0.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.label_smoothing = label_smoothing
        
        def forward(self, inputs, targets):
            num_classes = inputs.size(-1)
            if self.label_smoothing > 0:
                targets_one_hot = F.one_hot(targets, num_classes).float()
                targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                                  self.label_smoothing / num_classes
                log_probs = F.log_softmax(inputs, dim=-1)
                loss = -(targets_one_hot * log_probs).sum(dim=-1)
            else:
                loss = F.cross_entropy(inputs, targets, reduction='none')
            
            pt = torch.exp(-loss)
            focal_loss = ((1 - pt) ** self.gamma) * loss
            return focal_loss.mean()

    # ------------------------- 训练与测试函数 ------------------------------
    def train_epoch(model, optimizer, loss_fn, train_loader, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    def test_model(model, test_loader):
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def soft_voting(probs, box):
        n_samples = len(probs) // box
        probs = probs.reshape(n_samples, box, -1)
        avg_probs = probs.mean(axis=1)
        return avg_probs.argmax(axis=1)

    # ================== 计算详细分类指标函数 ==================
    def calculate_detailed_metrics(y_true, y_pred, num_classes=6):
        """
        计算每个类别的详细指标
        
        返回:
            dict: 包含各类别和整体指标的字典
        """
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        
        # 整体指标
        overall_acc = metrics.accuracy_score(y_true, y_pred)
        overall_mcc = metrics.matthews_corrcoef(y_true, y_pred)
        
        # 每个类别的指标
        class_metrics = []
        
        for class_idx in range(num_classes):
            # 二分类转换
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # 从混淆矩阵计算指标
            TP = conf_matrix[class_idx, class_idx]
            FN = conf_matrix[class_idx, :].sum() - TP
            FP = conf_matrix[:, class_idx].sum() - TP
            TN = conf_matrix.sum() - TP - FN - FP
            
            # Sensitivity (Recall / True Positive Rate)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            # Specificity (True Negative Rate)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            
            # Precision (Positive Predictive Value)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            
            # F1-score
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) \
                 if (precision + sensitivity) > 0 else 0.0
            
            # MCC for this class
            if len(np.unique(y_true_binary)) == 1 or len(np.unique(y_pred_binary)) == 1:
                class_mcc = 0.0
            else:
                class_mcc = metrics.matthews_corrcoef(y_true_binary, y_pred_binary)
            
            class_metrics.append({
                'class': class_idx,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1,
                'mcc': class_mcc,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN
            })
        
        return {
            'overall_accuracy': overall_acc,
            'overall_mcc': overall_mcc,
            'confusion_matrix': conf_matrix,
            'class_metrics': class_metrics
        }

    # --------------------------- 主循环 ------------------------------
    all_mcc = []
    all_acc = []
    loo = LeaveOneOut()
    n_samples = datX.shape[0]
    
    fold_predictions = []
    fold_true_labels = []
    
    for fold_idx, (train_index, test_index) in enumerate(loo.split(datX), start=1):
        train_X, test_X = datX[train_index, :], datX[test_index, :]
        train_y, test_y = ylab[train_index, :], ylab[test_index, :]

        print(f'种子 {seed} LOOCV: 样本 {fold_idx}/{n_samples}')
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed * 1000 + fold_idx)
        
        train_loader, test_loader, cur_box, ytrue = convert_data_dl(train_X, test_X, train_y, test_y)
        model = ImprovedNet(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
        loss_fn = FocalLoss(gamma=2, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        # 训练固定 Epochs
        for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, loss_fn, train_loader, epoch)
            scheduler.step()    
                       
        # 最终预测
        probs = test_model(model, test_loader)    

        # 软投票
        ytrue_0based = ytrue.reshape(1, cur_box)        
        preds_majority_0based_soft = soft_voting(probs, cur_box)[0]
        preds_majority_0based = preds_majority_0based_soft          
        ytrue_majority_0based = stats.mode(ytrue_0based, axis=1)[0].astype(int)[0]      
        
        # 转换回原始标签
        ytrue_majority = ytrue_majority_0based + 1
        preds_majority = preds_majority_0based + 1
        
        fold_predictions.append(preds_majority_0based)
        fold_true_labels.append(ytrue_majority_0based)
        
        print(f'样本 {fold_idx}: 真实={ytrue_majority}, 预测={preds_majority}')
        print('---' * 15)

    # ================== 计算详细性能指标 ==================
    fold_predictions = np.array(fold_predictions)
    fold_true_labels = np.array(fold_true_labels)
    
    detailed_metrics = calculate_detailed_metrics(fold_true_labels, fold_predictions, num_classes)
    
    # 计算平均类别MCC
    class_mccs = [m['mcc'] for m in detailed_metrics['class_metrics']]
    avg_class_mcc = np.mean(class_mccs)
    
    # 原始标签 (1-6)
    original_labels = [1, 2, 3, 4, 5, 6]
    
    # ================== 打印详细结果 ==================
    print(f'\n{"="*70}')
    print(f'种子 {seed} LOOCV 详细结果:')
    print(f'{"="*70}')
    
    print('\n【整体性能指标】')
    print(f'  Overall Accuracy (OA): {detailed_metrics["overall_accuracy"]:.4f}')
    print(f'  Overall MCC: {detailed_metrics["overall_mcc"]:.4f}')
    print(f'  Average Class MCC: {avg_class_mcc:.4f}')
    
    print('\n【混淆矩阵】')
    print(detailed_metrics['confusion_matrix'])
    
    print('\n【各类别详细指标】')
    print(f'{"Class":<8} {"Sens":<8} {"Spec":<8} {"Prec":<8} {"F1":<8} {"MCC":<8}')
    print('-' * 56)
    
    for i, metrics_dict in enumerate(detailed_metrics['class_metrics']):
        orig_label = original_labels[i]
        print(f'{orig_label:<8} '
              f'{metrics_dict["sensitivity"]:<8.4f} '
              f'{metrics_dict["specificity"]:<8.4f} '
              f'{metrics_dict["precision"]:<8.4f} '
              f'{metrics_dict["f1_score"]:<8.4f} '
              f'{metrics_dict["mcc"]:<8.4f}')
    
    print('\n【各类别混淆矩阵元素】')
    for i, metrics_dict in enumerate(detailed_metrics['class_metrics']):
        orig_label = original_labels[i]
        print(f'  类别 {orig_label}: TP={metrics_dict["TP"]}, '
              f'TN={metrics_dict["TN"]}, '
              f'FP={metrics_dict["FP"]}, '
              f'FN={metrics_dict["FN"]}')
    
    print('=' * 70)
    
    # 保存结果
    all_acc.append(detailed_metrics['overall_accuracy'])
    all_mcc.append({
        'overall_mcc': detailed_metrics['overall_mcc'],
        'class_mccs': class_mccs,
        'avg_class_mcc': avg_class_mcc
    })
    
    results.append({
        "seed": seed,
        "detailed_metrics": detailed_metrics,
        "all_mcc": all_mcc,
        "all_acc": all_acc
    })

end_time = time.time()
print(f'\n总运行时间：{(end_time - start_time)/60:.2f} 分钟')

# ========================= 最终结果汇总 =========================
print("\n" + "=" * 70)
print("最终结果汇总")
print("=" * 70)

# 收集所有种子的指标
all_oa = []
all_overall_mccs = []
all_avg_class_mccs = []
all_sensitivities = [[] for _ in range(num_classes)]
all_specificities = [[] for _ in range(num_classes)]
all_precisions = [[] for _ in range(num_classes)]
all_f1s = [[] for _ in range(num_classes)]
all_class_mccs_by_class = [[] for _ in range(num_classes)]

for result in results:
    dm = result['detailed_metrics']
    all_oa.append(dm['overall_accuracy'])
    all_overall_mccs.append(dm['overall_mcc'])
    all_avg_class_mccs.append(np.mean([m['mcc'] for m in dm['class_metrics']]))
    
    for i, class_metric in enumerate(dm['class_metrics']):
        all_sensitivities[i].append(class_metric['sensitivity'])
        all_specificities[i].append(class_metric['specificity'])
        all_precisions[i].append(class_metric['precision'])
        all_f1s[i].append(class_metric['f1_score'])
        all_class_mccs_by_class[i].append(class_metric['mcc'])

print("\n【整体性能】")
print(f"  Overall Accuracy (OA): {np.mean(all_oa):.4f} ± {np.std(all_oa):.4f}")
print(f"  Overall MCC: {np.mean(all_overall_mccs):.4f} ± {np.std(all_overall_mccs):.4f}")
print(f"  Average Class MCC: {np.mean(all_avg_class_mccs):.4f} ± {np.std(all_avg_class_mccs):.4f}")

print("\n【各类别Sensitivity】")
for i, orig_label in enumerate(original_labels):
    sens_values = all_sensitivities[i]
    print(f"  类别 {orig_label}: {np.mean(sens_values):.4f} ± {np.std(sens_values):.4f}")

print("\n【各类别Specificity】")
for i, orig_label in enumerate(original_labels):
    spec_values = all_specificities[i]
    print(f"  类别 {orig_label}: {np.mean(spec_values):.4f} ± {np.std(spec_values):.4f}")

print("\n【各类别Precision】")
for i, orig_label in enumerate(original_labels):
    prec_values = all_precisions[i]
    print(f"  类别 {orig_label}: {np.mean(prec_values):.4f} ± {np.std(prec_values):.4f}")

print("\n【各类别F1-score】")
for i, orig_label in enumerate(original_labels):
    f1_values = all_f1s[i]
    print(f"  类别 {orig_label}: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")

print("\n【各类别MCC】")
for i, orig_label in enumerate(original_labels):
    mcc_values = all_class_mccs_by_class[i]
    print(f"  类别 {orig_label}: {np.mean(mcc_values):.4f} ± {np.std(mcc_values):.4f}")

# 保存结果
save_dict = {
    'results': results,
    'summary': {
        'overall_accuracy': all_oa,
        'overall_mccs': all_overall_mccs,
        'avg_class_mccs': all_avg_class_mccs,
        'sensitivities_by_class': all_sensitivities,
        'specificities_by_class': all_specificities,
        'precisions_by_class': all_precisions,
        'f1_scores_by_class': all_f1s,
        'class_mccs_by_class': all_class_mccs_by_class,
        'class_labels': original_labels
    },
    'hyperparameters': {
        'epochs': epochs,
        'batch_size': bs,
        'dropout_rate': dropout_rate,
        'label_smoothing': label_smoothing,
        'weight_decay': weight_decay,
        'da': da,
        'num_classes': num_classes
    }
}

output_filename = f"ResNet_SV_LOOCV_CL317_uniref100_pssm20_r0Des_10mer_25gapcbd_Seed1to10_ep{epochs}_DA{da}_noES_ls01_dor05_wd005_bs512.npz"

np.savez(output_filename, **save_dict)
print(f"\n结果已保存至: {output_filename}")