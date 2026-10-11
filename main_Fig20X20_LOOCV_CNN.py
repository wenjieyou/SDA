# -*- coding: utf-8 -*-
"""
基础 CNN LOOCV - 精简版

WENJIE, Modified: 2025-12-07
"""

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

# ============================== 参数设置 ==============================
seeds = range(1, 6)  # 10个种子
results = []

for seed in seeds:
    print(f"\n{'='*60}\n种子 {seed}\n{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, epochs, img_size, da = 64, 15, 20, 50

    # ============================== 数据加载 ==============================
    data_file = 'PhosP_uniref50_1132_638_pssm20_r0Des_6mer_25gapcbd_DA150_DL00.npz'
    alldata = np.load(data_file)
    datX, ylab = alldata['datX'], alldata['ylab']
    
    datX = datX[:, :da, :, :]
    ylab = ylab[:, 0] if ylab.ndim > 1 else ylab
    box = datX.shape[1]
    n_samples = datX.shape[0]
    
    print(f"样本数: {n_samples}, 增强数: {box}, 总轮数: {n_samples * epochs:,}")

    # ============================== 数据处理 ==============================
    def convert_data_dl(train_X, test_X, train_y, test_y):
        n_train, n_test = train_X.shape[0], test_X.shape[0]
        
        train_X = train_X.reshape(n_train * box, 1, img_size, img_size).astype(np.float32)
        train_y = np.repeat(train_y, box).astype(np.int64)
        test_X = test_X.reshape(n_test * box, 1, img_size, img_size).astype(np.float32)
        test_y = np.repeat(test_y, box).astype(np.int64)
        
        train_loader = DataLoader(CNN_Dataset(train_X, train_y), batch_size=bs, shuffle=True)
        test_loader = DataLoader(CNN_Dataset(test_X, test_y), batch_size=bs, shuffle=False)
        
        return train_loader, test_loader, test_y

    class CNN_Dataset(Dataset):
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y)
        
        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]
        
        def __len__(self):
            return len(self.x_data)

    # ============================== 模型定义 ==============================
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, 5)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 5)
            self.bn2 = nn.BatchNorm2d(128)
            self.dropout = nn.Dropout2d()
            self.fc1 = nn.Linear(512, 100)
            self.fc2 = nn.Linear(100, 2)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
            x = F.relu(F.max_pool2d(self.dropout(self.bn2(self.conv2(x))), 2))
            x = x.view(-1, 512)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return self.fc2(x)

    # ============================== 训练/测试 ==============================
    def train_epoch(model, optimizer, loss_fn, loader):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        return total_loss / total, 100. * correct / total

    def test_model(model, loss_fn, loader):
        model.eval()
        total_loss, correct, total, all_preds = 0, 0, 0, []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                preds = outputs.max(1)[1]
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu().numpy())
        
        return np.concatenate(all_preds), total_loss / total, 100. * correct / total


    def safe_mode(arr):
        arr = np.asarray(arr).flatten()
        mode_result = stats.mode(arr, axis=None, keepdims=False)
        if hasattr(mode_result, 'mode'):
            return mode_result.mode.item()
        return np.asarray(mode_result[0]).item()

    # ============================== LOOCV主循环 ==============================
    all_predictions, all_true_labels = [], []
    loo = LeaveOneOut()
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(datX), 1):
        train_X, test_X = datX[train_idx], datX[test_idx]
        train_y, test_y = ylab[train_idx], ylab[test_idx]
        
        torch.manual_seed(seed * 10000 + fold)
        np.random.seed(seed * 10000 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 10000 + fold)
        
        train_loader, test_loader, ytrue = convert_data_dl(train_X, test_X, train_y, test_y)
        
        model = Net().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, loss_fn, train_loader)
            scheduler.step()
        
        preds, _, _ = test_model(model, loss_fn, test_loader)        
                       
        true_label = safe_mode(ytrue)
        pred_majority = safe_mode(preds)  
        
        all_predictions.append(pred_majority)
        all_true_labels.append(true_label)
        
        print(f"种子 {seed} | LOOCV进度: {fold}/{n_samples} 样本 {fold}: 真实={true_label}, 预测={pred_majority}")

    # ============================== 结果评估 ==============================
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    conf_matrix = metrics.confusion_matrix(all_true_labels, all_predictions)
    acc = metrics.accuracy_score(all_true_labels, all_predictions)
    mcc = metrics.matthews_corrcoef(all_true_labels, all_predictions)
    
    n_classes = len(np.unique(all_true_labels))
    avg_type = 'binary' if n_classes == 2 else 'macro'
    precision = metrics.precision_score(all_true_labels, all_predictions, average=avg_type)
    recall = metrics.recall_score(all_true_labels, all_predictions, average=avg_type)
    f1 = metrics.f1_score(all_true_labels, all_predictions, average=avg_type)
    
    # 计算敏感度和特异度
    if n_classes == 2:
        # 二分类：TN, FP, FN, TP
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感度 = 召回率 = TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度 = TNR
    else:
        # 多分类：计算每个类别的敏感度和特异度，然后取平均
        sensitivities = []
        specificities = []
        for i in range(n_classes):
            tp = conf_matrix[i, i]
            fn = conf_matrix[i, :].sum() - tp
            fp = conf_matrix[:, i].sum() - tp
            tn = conf_matrix.sum() - tp - fn - fp
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sens)
            specificities.append(spec)
        
        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
    
    print(f"\n种子 {seed} 结果:")
    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"Acc: {acc:.4f}, MCC: {mcc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    results.append({
        "seed": seed, "accuracy": acc, "mcc": mcc, "precision": precision,
        "recall": recall, "f1": f1, "sensitivity": sensitivity, "specificity": specificity,
        "confusion_matrix": conf_matrix,
        "predictions": all_predictions, "true_labels": all_true_labels
    })

# ============================== 最终汇总 ==============================
end_time = time.time()
total_time = (end_time - start_time) / 60

print(f"\n{'='*60}\n最终汇总\n{'='*60}")
print(f"总时间: {total_time:.2f}分钟, 平均每种子: {total_time/len(seeds):.2f}分钟")
print(f"实验数: {len(seeds)}种子 × {n_samples}样本 × {epochs}epochs = {len(seeds)*n_samples*epochs:,}轮\n")

all_acc = [r['accuracy'] for r in results]
all_mcc = [r['mcc'] for r in results]
all_prec = [r['precision'] for r in results]
all_rec = [r['recall'] for r in results]
all_f1 = [r['f1'] for r in results]
all_sens = [r['sensitivity'] for r in results]
all_spec = [r['specificity'] for r in results]

print(f"准确率:    {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"MCC:       {np.mean(all_mcc):.4f} ± {np.std(all_mcc):.4f}")
print(f"Precision: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
print(f"Recall:    {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
print(f"F1-Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"Sensitivity: {np.mean(all_sens):.4f} ± {np.std(all_sens):.4f}")
print(f"Specificity: {np.mean(all_spec):.4f} ± {np.std(all_spec):.4f}\n")

for r in results:
    print(f"种子{r['seed']:2d}: Acc={r['accuracy']:.4f}, MCC={r['mcc']:.4f}, "
          f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}, "
          f"Sens={r['sensitivity']:.4f}, Spec={r['specificity']:.4f}")

# ============================== 保存结果 ==============================
output_file = f"baseCNN_LOOCV_PhosP_uniref50_1132_638_pssm20_r0Des_6mer_25gapcbd_Seed12345_ep{epochs}_DA{da}_NoEarlyStop.npz" 
np.savez(output_file,
         results=results,
         summary={
             'mean_acc': np.mean(all_acc), 'std_acc': np.std(all_acc),
             'mean_mcc': np.mean(all_mcc), 'std_mcc': np.std(all_mcc),
             'mean_precision': np.mean(all_prec), 'std_precision': np.std(all_prec),
             'mean_recall': np.mean(all_rec), 'std_recall': np.std(all_rec),
             'mean_f1': np.mean(all_f1), 'std_f1': np.std(all_f1),
             'mean_sensitivity': np.mean(all_sens), 'std_sensitivity': np.std(all_sens),
             'mean_specificity': np.mean(all_spec), 'std_specificity': np.std(all_spec),
             'total_samples': n_samples,
             'total_training_epochs': len(seeds) * n_samples * epochs,
             'fixed_epochs': epochs,
             'early_stopping': False
         })
print(f"\n结果已保存: {output_file}")