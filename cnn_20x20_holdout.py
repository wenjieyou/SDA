# -*- coding: utf-8 -*-
"""
base CNN HoldOut

WENJIE, Modified: 2025-12-12
"""

import time
import numpy as np
from scipy import stats
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# Parameter Settings
seeds = range(1, 11)  # 10 seeds
results = []

for seed in seeds:
    print(f"\n{'='*60}\n Seed {seed}\n{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, epochs, img_size, da = 64, 15, 20, 50

    # Data Loading
    train_file = 'sda_dat/PDB1075_pssm20_6mer_25gapcbd_DA150_SDA.npz'
    test_file = 'sda_dat/PDB186_pssm20_6mer_25gapcbd_DA150_SDA.npz'
    
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    
    train_X, train_y = train_data['datX'], train_data['ylab']
    test_X, test_y = test_data['datX'], test_data['ylab']
    
    train_X = train_X[:, :da, :, :]
    test_X = test_X[:, :da, :, :]
    
    train_y = train_y[:, 0] if train_y.ndim > 1 else train_y
    test_y = test_y[:, 0] if test_y.ndim > 1 else test_y
    
    box = train_X.shape[1]
    n_train = train_X.shape[0]
    n_test = test_X.shape[0]
    
    print(f"Number of training samples: {n_train}, number of test samples: {n_test}, augmentation factor: {box}, total iterations: {n_train * epochs:,}")

    # Data processing
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

    # Model Definition
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

    # Training/Testing
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
        """Test the model and return prediction results, probabilities, and loss"""
        model.eval()
        total_loss, correct, total, all_preds, all_probs = 0, 0, 0, [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                # Obtain predicted probabilities
                probs = F.softmax(outputs, dim=1)
                
                total_loss += loss.item() * labels.size(0)
                preds = outputs.max(1)[1]
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_probs), total_loss / total, 100. * correct / total

    def safe_mode(arr):
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
        
        # Determine whether it is a binary classification task.
        n_classes = len(np.unique(ytrue))
        avg_type = 'binary' if n_classes == 2 else 'macro'
        
        precision = metrics.precision_score(ytrue, ypred, average=avg_type, zero_division=0)
        recall = metrics.recall_score(ytrue, ypred, average=avg_type, zero_division=0)
        f1 = metrics.f1_score(ytrue, ypred, average=avg_type, zero_division=0)
        
        # Sensitivity and Specificity (applicable only to binary classification)
        if n_classes == 2:
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
            
            # AUC 
            try:
                auc = metrics.roc_auc_score(ytrue, yprob[:, 1])
            except:
                auc = 0.0
        else:
            # Multi-class scenario
            sensitivity = recall  
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

    # Hold-out Training and Testing
    train_loader, test_loader, ytrue = convert_data_dl(train_X, test_X, train_y, test_y)
    
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training phase
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader)
        scheduler.step()
        # print(f"Seed {seed} | Epoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    
    # Testing phase
    preds, probs, test_loss, test_acc = test_model(model, loss_fn, test_loader)
    
    # Perform majority voting for each test sample
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    for i in range(n_test):
        sample_preds = preds[i * box:(i + 1) * box]
        sample_true = ytrue[i * box:(i + 1) * box]
        sample_probs = probs[i * box:(i + 1) * box]
        
        pred_majority = safe_mode(sample_preds)
        true_label = safe_mode(sample_true)
        prob_avg = np.mean(sample_probs, axis=0)  # Average the probabilities
        
        all_predictions.append(pred_majority)
        all_true_labels.append(true_label)
        all_probabilities.append(prob_avg)
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)

    # Outcome evaluation
    metrics_dict = calculate_metrics(all_true_labels, all_predictions, all_probabilities)
    
    print(f"\n Results for seed {seed}:")
    print(f"Confusion matrix:\n{metrics_dict['conf_matrix']}")
    print(f"Acc: {metrics_dict['acc']:.4f}, MCC: {metrics_dict['mcc']:.4f}, "
          f"P: {metrics_dict['precision']:.4f}, R: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}")
    print(f"Sensitivity: {metrics_dict['sensitivity']:.4f}, "
          f"Specificity: {metrics_dict['specificity']:.4f}, AUC: {metrics_dict['auc']:.4f}")
    
    results.append({
        "seed": seed,
        "accuracy": metrics_dict['acc'],
        "mcc": metrics_dict['mcc'],
        "precision": metrics_dict['precision'],
        "recall": metrics_dict['recall'],
        "f1": metrics_dict['f1'],
        "sensitivity": metrics_dict['sensitivity'],
        "specificity": metrics_dict['specificity'],
        "auc": metrics_dict['auc'],
        "confusion_matrix": metrics_dict['conf_matrix'],
        "predictions": all_predictions,
        "true_labels": all_true_labels,
        "probabilities": all_probabilities
    })

# Final Summary
end_time = time.time()
total_time = (end_time - start_time) / 60

print(f"\n{'='*60}\nFinal Summary\n{'='*60}")
print(f"Total time: {total_time:.2f}minutes, Average per seed: {total_time/len(seeds):.2f}minutes")
print(f"Experiment count: {len(seeds)} seeds × {n_train} samples × {epochs}epochs = {len(seeds)*n_train*epochs:,} rounds\n")

all_acc = [r['accuracy'] for r in results]
all_mcc = [r['mcc'] for r in results]
all_prec = [r['precision'] for r in results]
all_rec = [r['recall'] for r in results]
all_f1 = [r['f1'] for r in results]
all_sens = [r['sensitivity'] for r in results]
all_spec = [r['specificity'] for r in results]
all_auc = [r['auc'] for r in results]

print(f"Accuracy:    {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"MCC:         {np.mean(all_mcc):.4f} ± {np.std(all_mcc):.4f}")
print(f"Precision:   {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
print(f"Recall:      {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
print(f"F1-Score:    {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"Sensitivity: {np.mean(all_sens):.4f} ± {np.std(all_sens):.4f}")
print(f"Specificity: {np.mean(all_spec):.4f} ± {np.std(all_spec):.4f}")
print(f"AUC:         {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}\n")

for r in results:
    print(f"Seed {r['seed']:2d}: Acc={r['accuracy']:.4f}, MCC={r['mcc']:.4f}, "
          f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}, "
          f"Sen={r['sensitivity']:.4f}, Spe={r['specificity']:.4f}, AUC={r['auc']:.4f}")

# Save results
output_file = f"baseCNN_HO_PDB1075_PDB186_pssm20_6mer_25gapcbd_Seed1_{len(seeds)}_ep{epochs}_DA{da}_SDA.npz"
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
             'mean_auc': np.mean(all_auc), 'std_auc': np.std(all_auc),
             'train_samples': n_train,
             'test_samples': n_test,
             'total_training_epochs': len(seeds) * n_train * epochs,
             'fixed_epochs': epochs,
             'early_stopping': False,
             'validation_type': 'HoldOut'
         })
print(f"\nResults saved: {output_file}")