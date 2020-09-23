'''
@author: sahashumit
'''
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import pdb
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics

# Binary accuracy determination
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc 
    
# Load the model
import torchvision.models as models
model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Change the top layer for binary classification, you can add more layers here if necessary (See the Model Tuning script for different models)
# For Vgg19:
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),                   
                      nn.LogSoftmax(dim=1))
model.classifier

# model to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Define class weight if necessary. Otherwise use default. 
weights = [1.4, 0.2]
class_weights=torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters())

# Main Code
# Data loader and transformaiton    
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]),
        }

# Place the train and test file directories in the root (Check the organization folder text file for details)
train_dataset = datasets.ImageFolder(root ="",
                                      transform = image_transforms["train"]
                                     )   
test_dataset = datasets.ImageFolder(root ="",
                                      transform = image_transforms["test"]
                                     )  
# Make Train and Validation Split                                     
dataset_size = len(train_dataset)
dataset_indices = list(range(dataset_size))
np.random.shuffle(dataset_indices)
val_split_index = int(np.floor(0.1 * dataset_size)) # 10% validation
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# Train and Validation loader
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, sampler=val_sampler)

# Train Model
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
# no of epochs
n_epochs=20

for e in  range(n_epochs):

    # TRAINING
    i=0
    min_val_loss=np.Inf
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = binary_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = binary_acc(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    vLoss=val_epoch_loss/len(val_loader)
    
    # If the validation loss is at a minimum
    if vLoss < min_val_loss:
      # Save the model
      torch.save(model.state_dict(), "net.pth")  # OK
      epochs_no_improve = 0
      min_val_loss = vLoss
  
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | 
          Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| 
          Val Acc: {val_epoch_acc/len(val_loader):.3f}')

train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

# TEST model
# test loader
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
# load best model
model.load_state_dict(torch.load("net.pth"))
# TEST
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        y_test_pred = torch.log_softmax(y_test_pred, dim=1)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

# Evaluation

# Confusion Matrix, Accuracy, F1-Score
print(confusion_matrix(y_true_list, y_pred_list))
TN, FP, FN, TP = confusion_matrix(y_true_list,y_pred_list).ravel()
sensitivity = round((TP / (TP + FN)),4)
specificity = round((TN / (FP + TN)),4)
precision = round((TP / (TP + FP)),4)
recall = round((TP / (TP + FN)),4)
f1 = round(((2*precision*recall)/(precision+recall)),4)
if np.isnan(f1):
   f1 = 0
accuracy = round(((TP + TN) / len(y_true_list)),4)
from sklearn.metrics import roc_auc_score    
roc_auc_score(y_true_list, y_pred_list)   




