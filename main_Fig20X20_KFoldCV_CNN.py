# -*- coding: utf-8 -*-
"""
基础 CNN 10-Fold CV (10次重复) - 无 Early Stop版本
添加 Sensitivity, Specificity 和 AUC 指标

WENJIE, Modified: 2025-12-14
Enhanced with additional metrics: 2026-02-06
"""

from __future__ import print_function
import time
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# ============================== 参数设置 ==============================
seeds = range(1, 6)  # 10个种子
results = []  # 用于存储所有种子的结果

for seed in seeds:
    print(f"\n{'='*60}\n当前种子: {seed}\n{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, epochs, img_size, da = 64, 15, 20, 100

    # ============================== 数据加载 ============================== 
    data_file = 'metaiPVP/MetaiPVP_uniref50_TR_250_250_pssm20_r0Des_6mer_25gapcbd_DA150_DL00.npz'
    alldata = np.load(data_file)
    datX, ylab = alldata['datX'], alldata['ylab']
    
    datX = datX[:, :da, :, :]
    ylab = ylab[:, 0] if ylab.ndim > 1 else ylab
    box = datX.shape[1]
    n_samples = datX.shape[0]
    
    print(f"样本数: {n_samples}, 增强数: {box}, 每次实验总轮数: {n_samples * epochs:,}")

    # ============================== 数据处理 ==============================
    def convert_data_dl(train_X, test_X, train_y, test_y):
        """
        转换数据为深度学习训练需要的格式。
        将数据 reshape 为 (样本数 * box, 1, img_size, img_size)
        """
        n_train = train_X.shape[0]
        train_X = train_X.reshape(n_train * box, 1, img_size, img_size).astype(np.float32)
        train_y = np.repeat(train_y, box).astype(np.int64)
        
        n_test = test_X.shape[0]
        test_X = test_X.reshape(n_test * box, 1, img_size, img_size).astype(np.float32)
        test_y = np.repeat(test_y, box).astype(np.int64)
        
        train_dataset = CNN_Dataset(train_X, train_y)
        test_dataset = CNN_Dataset(test_X, test_y)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
        
        return train_loader, test_loader, test_y

    class CNN_Dataset(Dataset):
        """自定义 Dataset 类"""
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y)
        
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        
        def __len__(self):
            return len(self.x_data)

    # ============================== 模型定义 ==============================
    class Net(nn.Module):
        """CNN模型 - 直接返回logits用于CrossEntropyLoss"""
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

    # ============================== 训练/测试函数 ==============================
    def train_epoch(model, optimizer, loss_fn, loader, epoch):
        """训练一个epoch"""
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    def test_model(model, loss_fn, loader):
        """测试模型，返回预测结果、概率和损失"""
        model.eval()
        total_loss, corrects, all_preds, all_probs = 0.0, 0, [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                # 获取预测概率
                probs = F.softmax(outputs, dim=1)
                
                total_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                corrects += preds.eq(labels).sum().item()
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        accuracy = 100. * corrects / len(loader.dataset)
        print(f'Val loss: {avg_loss:.6f}, Accuracy: {corrects}/{len(loader.dataset)} ({accuracy:.4f}%)\n')
        
        return np.concatenate(all_preds), np.concatenate(all_probs), avg_loss
    
    def safe_mode(arr):
        """安全的众数计算,兼容不同scipy版本"""
        arr = np.asarray(arr).flatten()
        mode_result = stats.mode(arr, axis=None)
        if hasattr(mode_result, 'mode'):
            return mode_result.mode.item()
        return np.asarray(mode_result[0]).item()
    
    def calculate_metrics(ytrue, ypred, yprob):
        """
        计算所有评估指标
        
        参数:
            ytrue: 真实标签
            ypred: 预测标签
            yprob: 预测概率 (用于计算AUC)
        
        返回:
            包含所有指标的字典
        """
        # 混淆矩阵
        conf_matrix = metrics.confusion_matrix(ytrue, ypred)
        
        # 基本指标
        acc = metrics.accuracy_score(ytrue, ypred)
        mcc = metrics.matthews_corrcoef(ytrue, ypred)
        
        # 判断是否为二分类
        n_classes = len(np.unique(ytrue))
        avg_type = 'binary' if n_classes == 2 else 'macro'
        
        precision = metrics.precision_score(ytrue, ypred, average=avg_type, zero_division=0)
        recall = metrics.recall_score(ytrue, ypred, average=avg_type, zero_division=0)
        f1 = metrics.f1_score(ytrue, ypred, average=avg_type, zero_division=0)
        
        # Sensitivity 和 Specificity (仅适用于二分类)
        if n_classes == 2:
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 也叫 Recall 或 TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            
            # AUC (使用正类的概率)
            try:
                auc = metrics.roc_auc_score(ytrue, yprob[:, 1])
            except:
                auc = 0.0
        else:
            # 多分类情况
            sensitivity = recall  # 对于多分类，sensitivity就是recall
            specificity = 0.0  # 多分类没有单一的specificity
            try:
                auc = metrics.roc_auc_score(ytrue, yprob, multi_class='ovr', average='macro')
            except:
                auc = 0.0
        
        return {
            'conf_matrix': conf_matrix,
            'acc': acc,
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc
        }

    # ============================== 10-Fold CV主循环 ==============================
    rnd_time = 5  # 10次重复
    n_splits = 10  # 10折交叉验证
    
    all_mcc = []
    all_acc = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_auc = []

    for rnd in range(rnd_time):
        skf = StratifiedKFold(n_splits=n_splits, random_state=rnd, shuffle=True)
        fold_acc = []
        fold_mcc = []
        fold_precision = []
        fold_recall = []
        fold_f1 = []
        fold_sensitivity = []
        fold_specificity = []
        fold_auc = []
        
        for k, (train_index, test_index) in enumerate(skf.split(datX, ylab), start=1):
            train_X, test_X = datX[train_index], datX[test_index]
            train_y, test_y = ylab[train_index], ylab[test_index]

            print(f'种子 {seed} | 重复 {rnd+1}/{rnd_time} | Fold {k}/{n_splits}')
            print(f'训练集: {train_X.shape}, 测试集: {test_X.shape}')
            
            # 固定随机种子
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed * 1000 + rnd * 100 + k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 1000 + rnd * 100 + k)
            
            train_loader, test_loader, ytrue = convert_data_dl(train_X, test_X, train_y, test_y)
            
            # 初始化模型与优化器
            model = Net().to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # 训练与测试
            for epoch in range(1, epochs + 1):
                train_epoch(model, optimizer, loss_fn, train_loader, epoch)
                scheduler.step()

            preds, probs, val_loss = test_model(model, loss_fn, test_loader)
            
            # 评估: 对单个样本的 box 预测结果采用投票机制求众数
            n_test_samples = len(test_index)
            
            # 对每个测试样本分别求众数
            ytrue_majority = []
            preds_majority = []
            probs_majority = []
            
            for i in range(n_test_samples):
                # 提取第i个样本的box个预测/标签
                sample_ytrue = ytrue[i * box:(i + 1) * box]
                sample_preds = preds[i * box:(i + 1) * box]
                sample_probs = probs[i * box:(i + 1) * box]
                
                # 使用safe_mode求众数
                ytrue_majority.append(safe_mode(sample_ytrue))
                preds_majority.append(safe_mode(sample_preds))
                # 对概率取平均
                probs_majority.append(np.mean(sample_probs, axis=0))
            
            ytrue_majority = np.array(ytrue_majority)
            preds_majority = np.array(preds_majority)
            probs_majority = np.array(probs_majority)
            
            # 计算所有评估指标
            metrics_dict = calculate_metrics(ytrue_majority, preds_majority, probs_majority)
            
            print(f'Fold {k} 结果:')
            print(f'混淆矩阵:\n{metrics_dict["conf_matrix"]}')
            print(f'Acc: {metrics_dict["acc"]:.4f}, MCC: {metrics_dict["mcc"]:.4f}, '
                  f'P: {metrics_dict["precision"]:.4f}, R: {metrics_dict["recall"]:.4f}, F1: {metrics_dict["f1"]:.4f}')
            print(f'Sensitivity: {metrics_dict["sensitivity"]:.4f}, '
                  f'Specificity: {metrics_dict["specificity"]:.4f}, AUC: {metrics_dict["auc"]:.4f}')
            print('-----------------------------------\n')
            
            fold_acc.append(metrics_dict["acc"])
            fold_mcc.append(metrics_dict["mcc"])
            fold_precision.append(metrics_dict["precision"])
            fold_recall.append(metrics_dict["recall"])
            fold_f1.append(metrics_dict["f1"])
            fold_sensitivity.append(metrics_dict["sensitivity"])
            fold_specificity.append(metrics_dict["specificity"])
            fold_auc.append(metrics_dict["auc"])
        
        all_acc.append(fold_acc)
        all_mcc.append(fold_mcc)
        all_precision.append(fold_precision)
        all_recall.append(fold_recall)
        all_f1.append(fold_f1)
        all_sensitivity.append(fold_sensitivity)
        all_specificity.append(fold_specificity)
        all_auc.append(fold_auc)
        
        print(f'\n种子 {seed} | 重复 {rnd+1} 完成:')
        print(f'  平均 Acc: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}')
        print(f'  平均 MCC: {np.mean(fold_mcc):.4f} ± {np.std(fold_mcc):.4f}')
        print(f'  平均 AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}\n')

    # ============================== 种子结果汇总 ==============================
    seed_mean_acc = np.mean(all_acc)
    seed_mean_mcc = np.mean(all_mcc)
    seed_mean_precision = np.mean(all_precision)
    seed_mean_recall = np.mean(all_recall)
    seed_mean_f1 = np.mean(all_f1)
    seed_mean_sensitivity = np.mean(all_sensitivity)
    seed_mean_specificity = np.mean(all_specificity)
    seed_mean_auc = np.mean(all_auc)
    
    print(f'\n种子 {seed} 最终结果:')
    print(f'准确率:      {seed_mean_acc:.4f} ± {np.std(all_acc):.4f}')
    print(f'MCC:         {seed_mean_mcc:.4f} ± {np.std(all_mcc):.4f}')
    print(f'Precision:   {seed_mean_precision:.4f} ± {np.std(all_precision):.4f}')
    print(f'Recall:      {seed_mean_recall:.4f} ± {np.std(all_recall):.4f}')
    print(f'F1-Score:    {seed_mean_f1:.4f} ± {np.std(all_f1):.4f}')
    print(f'Sensitivity: {seed_mean_sensitivity:.4f} ± {np.std(all_sensitivity):.4f}')
    print(f'Specificity: {seed_mean_specificity:.4f} ± {np.std(all_specificity):.4f}')
    print(f'AUC:         {seed_mean_auc:.4f} ± {np.std(all_auc):.4f}')
    
    # 将当前种子的结果存储
    results.append({
        "seed": seed,
        "all_mcc": all_mcc,
        "all_acc": all_acc,
        "all_precision": all_precision,
        "all_recall": all_recall,
        "all_f1": all_f1,
        "all_sensitivity": all_sensitivity,
        "all_specificity": all_specificity,
        "all_auc": all_auc,
        "mean_acc": seed_mean_acc,
        "mean_mcc": seed_mean_mcc,
        "mean_precision": seed_mean_precision,
        "mean_recall": seed_mean_recall,
        "mean_f1": seed_mean_f1,
        "mean_sensitivity": seed_mean_sensitivity,
        "mean_specificity": seed_mean_specificity,
        "mean_auc": seed_mean_auc
    })

# ============================== 最终汇总 ==============================
end_time = time.time()
total_time = (end_time - start_time) / 60

print(f"\n{'='*60}\n最终结果汇总\n{'='*60}")
print(f"总时间: {total_time:.2f}分钟, 平均每种子: {total_time/len(seeds):.2f}分钟")
print(f"实验配置: {len(seeds)}个种子 × {rnd_time}次重复 × {n_splits}折 = {len(seeds) * rnd_time * n_splits}次训练")
print(f"每次训练: {epochs}个epochs\n")

# 跨种子的汇总统计
allmcc = [results[k]['mean_mcc'] for k in range(len(seeds))]
allacc = [results[k]['mean_acc'] for k in range(len(seeds))]
allprec = [results[k]['mean_precision'] for k in range(len(seeds))]
allrec = [results[k]['mean_recall'] for k in range(len(seeds))]
allf1 = [results[k]['mean_f1'] for k in range(len(seeds))]
allsens = [results[k]['mean_sensitivity'] for k in range(len(seeds))]
allspec = [results[k]['mean_specificity'] for k in range(len(seeds))]
allauc = [results[k]['mean_auc'] for k in range(len(seeds))]

print(f"跨种子平均准确率:      {np.mean(allacc):.4f} ± {np.std(allacc):.4f}")
print(f"跨种子平均MCC:         {np.mean(allmcc):.4f} ± {np.std(allmcc):.4f}")
print(f"跨种子平均Precision:   {np.mean(allprec):.4f} ± {np.std(allprec):.4f}")
print(f"跨种子平均Recall:      {np.mean(allrec):.4f} ± {np.std(allrec):.4f}")
print(f"跨种子平均F1-Score:    {np.mean(allf1):.4f} ± {np.std(allf1):.4f}")
print(f"跨种子平均Sensitivity: {np.mean(allsens):.4f} ± {np.std(allsens):.4f}")
print(f"跨种子平均Specificity: {np.mean(allspec):.4f} ± {np.std(allspec):.4f}")
print(f"跨种子平均AUC:         {np.mean(allauc):.4f} ± {np.std(allauc):.4f}\n")

print("各种子详细结果:")
for r in results:
    print(f"种子{r['seed']:2d}: Acc={r['mean_acc']:.4f}, MCC={r['mean_mcc']:.4f}, "
          f"P={r['mean_precision']:.4f}, R={r['mean_recall']:.4f}, F1={r['mean_f1']:.4f}, "
          f"Sen={r['mean_sensitivity']:.4f}, Spe={r['mean_specificity']:.4f}, AUC={r['mean_auc']:.4f}")

# ============================== 保存结果 ==============================
output_file = f"baseCNN_Results_MetaiPVP_uniref50_TR_250_250_pssm20_r0Des_6mer_25gapcbd_{rnd_time}X{n_splits}foldCV_Seed12345_ep{epochs}_DA{da}_NoES_bs{bs}.npz"
np.savez(output_file,
         results=results,
         summary={
             'mean_acc': np.mean(allacc), 'std_acc': np.std(allacc),
             'mean_mcc': np.mean(allmcc), 'std_mcc': np.std(allmcc),
             'mean_precision': np.mean(allprec), 'std_precision': np.std(allprec),
             'mean_recall': np.mean(allrec), 'std_recall': np.std(allrec),
             'mean_f1': np.mean(allf1), 'std_f1': np.std(allf1),
             'mean_sensitivity': np.mean(allsens), 'std_sensitivity': np.std(allsens),
             'mean_specificity': np.mean(allspec), 'std_specificity': np.std(allspec),
             'mean_auc': np.mean(allauc), 'std_auc': np.std(allauc),
             'n_seeds': len(seeds),
             'n_repeats': rnd_time,
             'n_folds': n_splits,
             'total_samples': n_samples,
             'fixed_epochs': epochs,
             'early_stopping': False
         })
print(f"\n结果已保存: {output_file}")