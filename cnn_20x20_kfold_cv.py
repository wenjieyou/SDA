# -*- coding: utf-8 -*-
"""
Basic CNN 10-Fold CV (10 repetitions) - Version without Early Stop

WENJIE, Modified: 2025-12-14

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

# Parameter Settings
seeds = range(1, 11)  # 10 seeds
results = []  # Used to store the results for all seeds

for seed in seeds:
    print(f"\n{'='*60}\nCurrent seeds: {seed}\n{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, epochs, img_size, da = 64, 15, 20, 100

    # Data Loading 
    data_file = 'sda_dat/MetaiPVP_TR_250_250_pssm20_6mer_25gapcbd_DA150_SDA.npz'
    alldata = np.load(data_file)
    datX, ylab = alldata['datX'], alldata['ylab']
    
    datX = datX[:, :da, :, :]
    ylab = ylab[:, 0] if ylab.ndim > 1 else ylab
    box = datX.shape[1]
    n_samples = datX.shape[0]
    
    print(f"Sample size: {n_samples}, Augmentation count: {box}, Total rounds per experiment: {n_samples * epochs:,}")

    # Data processing
    def convert_data_dl(train_X, test_X, train_y, test_y):
        """
        Convert the data into the format required for deep learning training. 
        Reshape the data to (number of samples * boxes, 1, img_size, img_size).
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
        """ Custom Dataset Class """
        def __init__(self, X, y):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y)
        
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        
        def __len__(self):
            return len(self.x_data)

    # Model Definition
    class Net(nn.Module):
        """CNN Model – Directly returns logits for CrossEntropyLoss"""
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

    # Training/Testing Functions
    def train_epoch(model, optimizer, loss_fn, loader, epoch):
        """Train for one epoch"""
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    def test_model(model, loss_fn, loader):
        """Test the model and return prediction results, probabilities, and loss"""
        model.eval()
        total_loss, corrects, all_preds, all_probs = 0.0, 0, [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                # Obtain predicted probabilities
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
        """Mode calculation"""
        arr = np.asarray(arr).flatten()
        mode_result = stats.mode(arr, axis=None)
        if hasattr(mode_result, 'mode'):
            return mode_result.mode.item()
        return np.asarray(mode_result[0]).item()
    
    def calculate_metrics(ytrue, ypred, yprob):
        """
        Calculate all evaluation metrics
			Parameters:
				ytrue: Ground truth labels
				ypred: Predicted labels
				yprob: Predicted probabilities (used to calculate AUC)
			Returns:
				A dictionary containing all metrics
        """
        # Confusion matrix
        conf_matrix = metrics.confusion_matrix(ytrue, ypred)
        
        # Basic indicators
        acc = metrics.accuracy_score(ytrue, ypred)
        mcc = metrics.matthews_corrcoef(ytrue, ypred)
        
        # Determine whether it is a binary classification task
        n_classes = len(np.unique(ytrue))
        avg_type = 'binary' if n_classes == 2 else 'macro'
        
        precision = metrics.precision_score(ytrue, ypred, average=avg_type, zero_division=0)
        recall = metrics.recall_score(ytrue, ypred, average=avg_type, zero_division=0)
        f1 = metrics.f1_score(ytrue, ypred, average=avg_type, zero_division=0)
        
        # Sensitivity and Specificity (only to binary classification)
        if n_classes == 2:
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall or TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            
            # AUC (Probability of the positive class)
            try:
                auc = metrics.roc_auc_score(ytrue, yprob[:, 1])
            except:
                auc = 0.0
        else:
            # Multi-class scenario
            sensitivity = recall  # For multi-class classification, sensitivity is equivalent to recall
            specificity = 0.0  
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

    # 10-fold CV main loop
    rnd_time = 5  # 10 repetitions
    n_splits = 10  # 10-fold cross-validation
    
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

            print(f'Seed {seed} | Repeat {rnd+1}/{rnd_time} | Fold {k}/{n_splits}')
            print(f'Training set: {train_X.shape}, Test set: {test_X.shape}')
            
            # Fix the random seed
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed * 1000 + rnd * 100 + k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 1000 + rnd * 100 + k)
            
            train_loader, test_loader, ytrue = convert_data_dl(train_X, test_X, train_y, test_y)
            
            # Initialize the model and optimizer
            model = Net().to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # Training and Testing
            for epoch in range(1, epochs + 1):
                train_epoch(model, optimizer, loss_fn, train_loader, epoch)
                scheduler.step()

            preds, probs, val_loss = test_model(model, loss_fn, test_loader)
            
            # Evaluation: A voting mechanism is used to determine the mode of the BOX prediction results for individual samples
            n_test_samples = len(test_index)
            
            # Calculate the mode for each test sample individually
            ytrue_majority = []
            preds_majority = []
            probs_majority = []
            
            for i in range(n_test_samples):
                # Extract the box predictions/labels for the i-th sample.
                sample_ytrue = ytrue[i * box:(i + 1) * box]
                sample_preds = preds[i * box:(i + 1) * box]
                sample_probs = probs[i * box:(i + 1) * box]
                
                # Calculate the mode using safe_mode
                ytrue_majority.append(safe_mode(sample_ytrue))
                preds_majority.append(safe_mode(sample_preds))
                # Average the probabilities
                probs_majority.append(np.mean(sample_probs, axis=0))
            
            ytrue_majority = np.array(ytrue_majority)
            preds_majority = np.array(preds_majority)
            probs_majority = np.array(probs_majority)
            
            # Calculate all evaluation metrics
            metrics_dict = calculate_metrics(ytrue_majority, preds_majority, probs_majority)
            
            print(f'Fold {k} result:')
            print(f'Confusion matrix:\n{metrics_dict["conf_matrix"]}')
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
        
        print(f'\nSeed {seed} | Repeat {rnd+1} Completed:')
        print(f'  Average Acc: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}')
        print(f'  Average MCC: {np.mean(fold_mcc):.4f} ± {np.std(fold_mcc):.4f}')
        print(f'  Average AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}\n')

    # Summary of results for each seed
    seed_mean_acc = np.mean(all_acc)
    seed_mean_mcc = np.mean(all_mcc)
    seed_mean_precision = np.mean(all_precision)
    seed_mean_recall = np.mean(all_recall)
    seed_mean_f1 = np.mean(all_f1)
    seed_mean_sensitivity = np.mean(all_sensitivity)
    seed_mean_specificity = np.mean(all_specificity)
    seed_mean_auc = np.mean(all_auc)
    
    print(f'\nSeed {seed} Final result:')
    print(f'Accuracy:      {seed_mean_acc:.4f} ± {np.std(all_acc):.4f}')
    print(f'MCC:         {seed_mean_mcc:.4f} ± {np.std(all_mcc):.4f}')
    print(f'Precision:   {seed_mean_precision:.4f} ± {np.std(all_precision):.4f}')
    print(f'Recall:      {seed_mean_recall:.4f} ± {np.std(all_recall):.4f}')
    print(f'F1-Score:    {seed_mean_f1:.4f} ± {np.std(all_f1):.4f}')
    print(f'Sensitivity: {seed_mean_sensitivity:.4f} ± {np.std(all_sensitivity):.4f}')
    print(f'Specificity: {seed_mean_specificity:.4f} ± {np.std(all_specificity):.4f}')
    print(f'AUC:         {seed_mean_auc:.4f} ± {np.std(all_auc):.4f}')
    
    # Store the results of the current seed
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

# Final Summary
end_time = time.time()
total_time = (end_time - start_time) / 60

print(f"\n{'='*60}\nSummary of Final Results\n{'='*60}")
print(f"Total time: {total_time:.2f}minutes, average per seed: {total_time/len(seeds):.2f}minutes")
print(f"Experimental setup: {len(seeds)}seeds × {rnd_time}repetitions × {n_splits}folds = {len(seeds) * rnd_time * n_splits}training runs")
print(f"Each training run: {epochs} epochs\n")

# Cross-seed summary statistics
allmcc = [results[k]['mean_mcc'] for k in range(len(seeds))]
allacc = [results[k]['mean_acc'] for k in range(len(seeds))]
allprec = [results[k]['mean_precision'] for k in range(len(seeds))]
allrec = [results[k]['mean_recall'] for k in range(len(seeds))]
allf1 = [results[k]['mean_f1'] for k in range(len(seeds))]
allsens = [results[k]['mean_sensitivity'] for k in range(len(seeds))]
allspec = [results[k]['mean_specificity'] for k in range(len(seeds))]
allauc = [results[k]['mean_auc'] for k in range(len(seeds))]

print(f"Cross-seed average accuracy:    {np.mean(allacc):.4f} ± {np.std(allacc):.4f}")
print(f"Cross-seed average MCC:         {np.mean(allmcc):.4f} ± {np.std(allmcc):.4f}")
print(f"Cross-seed average Precision:   {np.mean(allprec):.4f} ± {np.std(allprec):.4f}")
print(f"Cross-seed average Recall:      {np.mean(allrec):.4f} ± {np.std(allrec):.4f}")
print(f"Cross-seed average F1-Score:    {np.mean(allf1):.4f} ± {np.std(allf1):.4f}")
print(f"Cross-seed average Sensitivity: {np.mean(allsens):.4f} ± {np.std(allsens):.4f}")
print(f"Cross-seed average Specificity: {np.mean(allspec):.4f} ± {np.std(allspec):.4f}")
print(f"Cross-seed average AUC:         {np.mean(allauc):.4f} ± {np.std(allauc):.4f}\n")

print("Detailed results for each seed:")
for r in results:
    print(f"Seed{r['seed']:2d}: Acc={r['mean_acc']:.4f}, MCC={r['mean_mcc']:.4f}, "
          f"P={r['mean_precision']:.4f}, R={r['mean_recall']:.4f}, F1={r['mean_f1']:.4f}, "
          f"Sen={r['mean_sensitivity']:.4f}, Spe={r['mean_specificity']:.4f}, AUC={r['mean_auc']:.4f}")

# Save results
output_file = f"baseCNN_Results_MetaiPVP_pssm20_6mer_25gapcbd_{rnd_time}X{n_splits}foldCV_Seed1_10_ep{epochs}_DA{da}_SDA.npz"
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
print(f"\nResults saved: {output_file}")