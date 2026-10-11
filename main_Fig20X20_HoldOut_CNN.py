# -*- coding: utf-8 -*-
"""
Multi_Dataset_CNN_Ensemble_with_Majority_Voting
使用10个平衡子集训练10个CNN分类器，并通过多数投票集成预测结果

修改说明：
1. 使用10个平衡子集（CBalanced0-9）分别训练10个独立的CNN分类器
2. 每个分类器对测试集进行预测
3. 使用多数投票方法集成10个分类器的预测结果
4. 评估集成模型的性能

WENJIE: 2025-01-27
"""

from __future__ import print_function
import time
import numpy as np
from scipy import stats
from sklearn import metrics
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# ------------------------------ 参数设置 ------------------------------
num_subsets = 10  # 10个平衡子集（CBalanced0 到 CBalanced9）
seeds = range(1, 11)  # 10个种子
rnd_time = 10  # 每个种子重复运行10次

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 128  # batch_size
epochs = 15  # 固定训练轮数
img_size = 20  # 图片的宽高
da = 50  # 数据增强参数

# 测试集文件（所有子集共享同一个测试集）
test_file = 'PaCRISPR_IND_uniref90/PaCRISPR_uniref90_IND_26_26_pssm20_CBalanced0_r0Des_20mer_25gapcbd_DA500_DL00.npz'

# ------------------------------ 工具函数 ------------------------------
def safe_mode(arr):
    """安全的众数计算函数"""
    arr = np.asarray(arr).flatten()
    mode_result = stats.mode(arr, axis=None)
    if hasattr(mode_result, 'mode'):
        return mode_result.mode.item()
    return np.asarray(mode_result[0]).item()

def majority_vote(predictions_list):
    """
    对多个分类器的预测结果进行多数投票
    
    参数:
        predictions_list: list of arrays, 每个元素是一个分类器的预测结果数组
    
    返回:
        ensemble_predictions: 集成后的预测结果
    """
    predictions_array = np.array(predictions_list)  # shape: (num_classifiers, num_samples)
    ensemble_predictions = np.array([safe_mode(predictions_array[:, i]) for i in range(predictions_array.shape[1])])
    return ensemble_predictions

# ------------------------------ 数据集类 ------------------------------
class CNN_Dataset(Dataset):
    def __init__(self, X, y):
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return len(self.x_data)

def convert_data_dl(train_X, test_X, train_y, test_y, box):
    """转换数据为DataLoader格式"""
    n_train, n_test = train_X.shape[0], test_X.shape[0]
    
    train_X = train_X.reshape(n_train * box, 1, img_size, img_size).astype(np.float32)
    train_y = np.repeat(train_y, box).astype(np.int64)
    test_X = test_X.reshape(n_test * box, 1, img_size, img_size).astype(np.float32)
    test_y = np.repeat(test_y, box).astype(np.int64)
    
    train_loader = DataLoader(CNN_Dataset(train_X, train_y), batch_size=bs, shuffle=True)
    test_loader = DataLoader(CNN_Dataset(test_X, test_y), batch_size=bs, shuffle=False)
    
    return train_loader, test_loader, test_y

# ------------------------------ CNN模型定义 ------------------------------
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

# ------------------------------ 训练和测试函数 ------------------------------
def train_epoch(model, optimizer, loss_fn, loader):
    """训练一个epoch"""
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
    """测试模型"""
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

def calculate_metrics(ytrue, ypred):
    """计算所有评估指标"""
    conf_matrix = metrics.confusion_matrix(ytrue, ypred)
    acc = metrics.accuracy_score(ytrue, ypred)
    mcc = metrics.matthews_corrcoef(ytrue, ypred)
    
    n_classes = len(np.unique(ytrue))
    avg_type = 'binary' if n_classes == 2 else 'macro'
    precision = metrics.precision_score(ytrue, ypred, average=avg_type)
    recall = metrics.recall_score(ytrue, ypred, average=avg_type)
    f1 = metrics.f1_score(ytrue, ypred, average=avg_type)
    
    # 计算敏感性(SN)和特异性(SP)
    if n_classes == 2:
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = recall
        specificity = 0
        for i in range(n_classes):
            tn = np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i])
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificity += tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity /= n_classes
    
    return {
        'conf_matrix': conf_matrix,
        'acc': acc,
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

# ------------------------------ 主程序 ------------------------------
print("="*80)
print("使用10个平衡子集训练CNN分类器并进行多数投票集成")
print("="*80)

# 检查测试集文件是否存在
if not os.path.exists(test_file):
    raise FileNotFoundError(f"测试数据文件 {test_file} 不存在")

# 加载测试数据（只需加载一次）
test_data = np.load(test_file)
test_X = test_data['datX']
test_y = test_data['ylab']
test_X = test_X[:, :da, :, :]
test_y = test_y[:, 0] if test_y.ndim > 1 else test_y

print(f'测试集形状: {test_X.shape}')
print(f'测试集标签分布: {np.bincount(test_y.astype(int))}')

# 存储所有种子和重复的结果
all_seeds_results = []

for seed in seeds:
    print(f"\n{'='*80}")
    print(f"处理种子: {seed}")
    print(f"{'='*80}")
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 存储当前种子所有重复的结果
    seed_results = {
        'individual_acc': [],  # 单个分类器的准确率
        'ensemble_acc': [],    # 集成后的准确率
        'individual_mcc': [],
        'ensemble_mcc': [],
        'individual_precision': [],
        'ensemble_precision': [],
        'individual_recall': [],
        'ensemble_recall': [],
        'individual_f1': [],
        'ensemble_f1': [],
        'individual_sensitivity': [],
        'ensemble_sensitivity': [],
        'individual_specificity': [],
        'ensemble_specificity': []
    }
    
    for rnd in range(rnd_time):
        print(f'\n--- 种子 {seed}, 重复 {rnd+1}/{rnd_time} ---')
        
        # 设置当前重复的随机种子
        current_seed = seed + rnd
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        
        # 存储10个分类器的预测结果
        all_predictions = []
        individual_metrics_list = []
        
        # 训练10个分类器（使用10个平衡子集）
        for subset_idx in range(num_subsets):
            print(f'\n  训练分类器 {subset_idx+1}/{num_subsets} (CBalanced{subset_idx})...')
            
            # 加载训练数据
            train_file = f'PaCRISPR_TRN_uniref90/PaCRISPR_uniref90_TRN_98_98_pssm20_CBalanced{subset_idx}_r0Des_top20mercovgap0_24cityblock_DA500_DL00.npz'
            
            if not os.path.exists(train_file):
                print(f"  警告: 训练文件 {train_file} 不存在，跳过此子集")
                continue
            
            train_data = np.load(train_file)
            train_X = train_data['datX']
            train_y = train_data['ylab']
            train_X = train_X[:, :da, :, :]
            train_y = train_y[:, 0] if train_y.ndim > 1 else train_y
            
            box = train_X.shape[1]
            
            # 准备数据
            train_loader, test_loader, ytrue = convert_data_dl(train_X, test_X, train_y, test_y, box)
            
            # 初始化模型
            model = Net().to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # 训练模型
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader)
                scheduler.step()
                if epoch == 1 or epoch == epochs:
                    print(f"    Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            
            # 测试模型
            preds, test_loss, test_acc = test_model(model, loss_fn, test_loader)
            
            # 对单个样本的box预测结果采用投票机制
            n_test_samples = test_y.shape[0]
            ytrue_majority = np.array([safe_mode(ytrue[i*box:(i+1)*box]) for i in range(n_test_samples)])
            preds_majority = np.array([safe_mode(preds[i*box:(i+1)*box]) for i in range(n_test_samples)])
            
            # 保存预测结果用于后续集成
            all_predictions.append(preds_majority)
            
            # 计算单个分类器的指标
            individual_metrics = calculate_metrics(ytrue_majority, preds_majority)
            individual_metrics_list.append(individual_metrics)
            
            print(f"    分类器 {subset_idx+1} - Acc: {individual_metrics['acc']:.4f}, "
                  f"MCC: {individual_metrics['mcc']:.4f}, F1: {individual_metrics['f1']:.4f}")
        
        # 如果没有成功训练任何分类器，跳过此重复
        if len(all_predictions) == 0:
            print(f"  警告: 重复 {rnd+1} 没有成功训练任何分类器，跳过")
            continue
        
        # 多数投票集成
        print(f'\n  执行多数投票集成 ({len(all_predictions)} 个分类器)...')
        ensemble_predictions = majority_vote(all_predictions)
        
        # 计算集成模型的指标
        ensemble_metrics = calculate_metrics(ytrue_majority, ensemble_predictions)
        
        print(f'\n  === 重复 {rnd+1} 结果汇总 ===')
        print(f'  单个分类器平均性能:')
        print(f'    Acc: {np.mean([m["acc"] for m in individual_metrics_list]):.4f} ± '
              f'{np.std([m["acc"] for m in individual_metrics_list]):.4f}')
        print(f'    MCC: {np.mean([m["mcc"] for m in individual_metrics_list]):.4f} ± '
              f'{np.std([m["mcc"] for m in individual_metrics_list]):.4f}')
        print(f'    F1:  {np.mean([m["f1"] for m in individual_metrics_list]):.4f} ± '
              f'{np.std([m["f1"] for m in individual_metrics_list]):.4f}')
        
        print(f'\n  集成模型性能:')
        print(f'    Confusion Matrix:\n{ensemble_metrics["conf_matrix"]}')
        print(f'    Acc: {ensemble_metrics["acc"]:.4f}')
        print(f'    MCC: {ensemble_metrics["mcc"]:.4f}')
        print(f'    Precision: {ensemble_metrics["precision"]:.4f}')
        print(f'    Recall: {ensemble_metrics["recall"]:.4f}')
        print(f'    F1: {ensemble_metrics["f1"]:.4f}')
        print(f'    Sensitivity(SN): {ensemble_metrics["sensitivity"]:.4f}')
        print(f'    Specificity(SP): {ensemble_metrics["specificity"]:.4f}')
        
        # 保存结果
        seed_results['individual_acc'].append(np.mean([m["acc"] for m in individual_metrics_list]))
        seed_results['ensemble_acc'].append(ensemble_metrics['acc'])
        seed_results['individual_mcc'].append(np.mean([m["mcc"] for m in individual_metrics_list]))
        seed_results['ensemble_mcc'].append(ensemble_metrics['mcc'])
        seed_results['individual_precision'].append(np.mean([m["precision"] for m in individual_metrics_list]))
        seed_results['ensemble_precision'].append(ensemble_metrics['precision'])
        seed_results['individual_recall'].append(np.mean([m["recall"] for m in individual_metrics_list]))
        seed_results['ensemble_recall'].append(ensemble_metrics['recall'])
        seed_results['individual_f1'].append(np.mean([m["f1"] for m in individual_metrics_list]))
        seed_results['ensemble_f1'].append(ensemble_metrics['f1'])
        seed_results['individual_sensitivity'].append(np.mean([m["sensitivity"] for m in individual_metrics_list]))
        seed_results['ensemble_sensitivity'].append(ensemble_metrics['sensitivity'])
        seed_results['individual_specificity'].append(np.mean([m["specificity"] for m in individual_metrics_list]))
        seed_results['ensemble_specificity'].append(ensemble_metrics['specificity'])
    
    # 汇总当前种子的结果
    print(f'\n{"="*80}')
    print(f'种子 {seed} 最终结果汇总 ({len(seed_results["ensemble_acc"])} 次重复)')
    print(f'{"="*80}')
    print(f'单个分类器平均性能:')
    print(f'  Acc: {np.mean(seed_results["individual_acc"]):.4f} ± {np.std(seed_results["individual_acc"]):.4f}')
    print(f'  MCC: {np.mean(seed_results["individual_mcc"]):.4f} ± {np.std(seed_results["individual_mcc"]):.4f}')
    print(f'  F1:  {np.mean(seed_results["individual_f1"]):.4f} ± {np.std(seed_results["individual_f1"]):.4f}')
    print(f'\n集成模型性能:')
    print(f'  Acc: {np.mean(seed_results["ensemble_acc"]):.4f} ± {np.std(seed_results["ensemble_acc"]):.4f}')
    print(f'  MCC: {np.mean(seed_results["ensemble_mcc"]):.4f} ± {np.std(seed_results["ensemble_mcc"]):.4f}')
    print(f'  Precision: {np.mean(seed_results["ensemble_precision"]):.4f} ± {np.std(seed_results["ensemble_precision"]):.4f}')
    print(f'  Recall: {np.mean(seed_results["ensemble_recall"]):.4f} ± {np.std(seed_results["ensemble_recall"]):.4f}')
    print(f'  F1: {np.mean(seed_results["ensemble_f1"]):.4f} ± {np.std(seed_results["ensemble_f1"]):.4f}')
    print(f'  Sensitivity(SN): {np.mean(seed_results["ensemble_sensitivity"]):.4f} ± {np.std(seed_results["ensemble_sensitivity"]):.4f}')
    print(f'  Specificity(SP): {np.mean(seed_results["ensemble_specificity"]):.4f} ± {np.std(seed_results["ensemble_specificity"]):.4f}')
    
    all_seeds_results.append(seed_results)

# ------------------------------ 最终汇总 ------------------------------
print(f'\n{"="*80}')
print(f'所有种子最终汇总 ({len(seeds)} 个种子 × {rnd_time} 次重复)')
print(f'{"="*80}')

# 合并所有种子的结果
final_individual_acc = []
final_ensemble_acc = []
final_individual_mcc = []
final_ensemble_mcc = []
final_individual_f1 = []
final_ensemble_f1 = []
final_ensemble_precision = []
final_ensemble_recall = []
final_ensemble_sensitivity = []
final_ensemble_specificity = []

for seed_results in all_seeds_results:
    final_individual_acc.extend(seed_results['individual_acc'])
    final_ensemble_acc.extend(seed_results['ensemble_acc'])
    final_individual_mcc.extend(seed_results['individual_mcc'])
    final_ensemble_mcc.extend(seed_results['ensemble_mcc'])
    final_individual_f1.extend(seed_results['individual_f1'])
    final_ensemble_f1.extend(seed_results['ensemble_f1'])
    final_ensemble_precision.extend(seed_results['ensemble_precision'])
    final_ensemble_recall.extend(seed_results['ensemble_recall'])
    final_ensemble_sensitivity.extend(seed_results['ensemble_sensitivity'])
    final_ensemble_specificity.extend(seed_results['ensemble_specificity'])

print(f'单个分类器平均性能:')
print(f'  Acc: {np.mean(final_individual_acc):.4f} ± {np.std(final_individual_acc):.4f}')
print(f'  MCC: {np.mean(final_individual_mcc):.4f} ± {np.std(final_individual_mcc):.4f}')
print(f'  F1:  {np.mean(final_individual_f1):.4f} ± {np.std(final_individual_f1):.4f}')

print(f'\n集成模型性能:')
print(f'  Acc: {np.mean(final_ensemble_acc):.4f} ± {np.std(final_ensemble_acc):.4f}')
print(f'  MCC: {np.mean(final_ensemble_mcc):.4f} ± {np.std(final_ensemble_mcc):.4f}')
print(f'  Precision: {np.mean(final_ensemble_precision):.4f} ± {np.std(final_ensemble_precision):.4f}')
print(f'  Recall: {np.mean(final_ensemble_recall):.4f} ± {np.std(final_ensemble_recall):.4f}')
print(f'  F1: {np.mean(final_ensemble_f1):.4f} ± {np.std(final_ensemble_f1):.4f}')
print(f'  Sensitivity(SN): {np.mean(final_ensemble_sensitivity):.4f} ± {np.std(final_ensemble_sensitivity):.4f}')
print(f'  Specificity(SP): {np.mean(final_ensemble_specificity):.4f} ± {np.std(final_ensemble_specificity):.4f}')

print(f'\n性能提升:')
acc_improvement = np.mean(final_ensemble_acc) - np.mean(final_individual_acc)
mcc_improvement = np.mean(final_ensemble_mcc) - np.mean(final_individual_mcc)
f1_improvement = np.mean(final_ensemble_f1) - np.mean(final_individual_f1)
print(f'  Acc 提升: {acc_improvement:+.4f}')
print(f'  MCC 提升: {mcc_improvement:+.4f}')
print(f'  F1 提升:  {f1_improvement:+.4f}')

# 保存结果
end_time = time.time()
total_time = (end_time - start_time) / 60

print(f'\n总运行时间: {total_time:.2f} 分钟')

# 保存所有结果
output_filename = f"EnsVot_HO_PaCRISPR_uniref90_TR2025_9898_Ind2025_2626_{num_subsets}Subsets_{rnd_time}Repeats_Seed1to{len(seeds)}_ep{epochs}_DA{da}_bs{bs}_CB0.npz"
np.savez(output_filename, 
         all_seeds_results=all_seeds_results,
         final_individual_acc=final_individual_acc,
         final_ensemble_acc=final_ensemble_acc,
         final_individual_mcc=final_individual_mcc,
         final_ensemble_mcc=final_ensemble_mcc,
         final_individual_f1=final_individual_f1,
         final_ensemble_f1=final_ensemble_f1,
         final_ensemble_precision=final_ensemble_precision,
         final_ensemble_recall=final_ensemble_recall,
         final_ensemble_sensitivity=final_ensemble_sensitivity,
         final_ensemble_specificity=final_ensemble_specificity,
         num_subsets=num_subsets,
         num_seeds=len(seeds),
         rnd_time=rnd_time,
         epochs=epochs,
         da=da)

print(f'结果已保存到: {output_filename}')
print(f'\n{"="*80}')
print("程序执行完成!")
print(f'{"="*80}')
