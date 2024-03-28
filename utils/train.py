import os
import glob
from tqdm import tqdm
import pandas as pd
import time 

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from torchvision.io import read_image
import torchvision.models as models
from torch_poly_lr_decay import PolynomialLRDecay

from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch import Tensor
import torch.optim as optim
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet


import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score

import copy
import pickle

import utils


def train_epoch(model, dataloaders, criterion, optimizer, epoch):
    num_epochs = 30
    train_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for ii, (inputs, labels) in enumerate(dataloaders):

        inputs= inputs.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        output = model(inputs)


        # Loss and backpropagation of gradients
        loss = criterion(output,labels)
        loss.backward()

        # Update the parameters
        optimizer.step()

        #scheduler.step()

        # total
        total += labels.size(0)

        # Track train loss by multiplying average loss by number of examples in batch
        train_loss += loss.item() * inputs.size(0)

        # Calculate accuracy by finding max log probability
        _, pred = torch.max(output, dim=1)
        correct += pred.eq(labels).sum().item()

        print(f"epoch: {epoch+1}/{num_epochs}, repeat: {ii}", end='\r')


    train_loss = train_loss/total
    train_acc = correct/total

    return train_loss, train_acc


#### Validation ####
def valid_epoch(model, dataloaders, criterion):
    valid_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    all_labels =[]
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloaders:

            # Tensors to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(inputs)

            # Validation loss
            loss = criterion(output, labels)

            # total
            total += labels.size(0)

            # Multiply average loss times the number of examples in batch
            valid_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct += pred.eq(labels).sum().item()

            # Calculate AUC
            all_labels.extend(labels.cpu().detach().numpy().tolist())
            all_predictions.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())


    # Cacluate all AUC
    all_y_pred_labels = np.array(all_predictions)[:,1]
    val_roc_auc = roc_auc_score(all_labels, all_y_pred_labels)

    valid_loss = valid_loss/total
    valid_acc = correct/total

    return valid_loss, valid_acc, val_roc_auc


def run_train(train_df, model_name, device,otpath ):

    ## set parameters
    lr = 1e-5
    batch_size = 256
    num_epochs = 30
    num_workers = 21

    if not os.path.exists(otpath):
        os.makedirs(otpath)


    ### Binairzation ###
    le = preprocessing.LabelEncoder()
    train_df['Status'] = le.fit_transform(train_df['Status'])

    print("#train----")
    print(train_df['Status'].value_counts())


    for fold  in range(0,5):

        print("============")
        print(f'FOLD {fold + 1}')
        print("============")
        validation_data = train_df.loc[train_df['id'] == fold+1]
        training_data = train_df.loc[train_df['id'] != fold+1]

        train_data = CustomImageDataset_from_csv(training_data ,  transform = image_transform['train'])
        train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = False ,num_workers=num_workers)

        val_data = CustomImageDataset_from_csv(validation_data ,  transform = image_transform['val'])
        val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = False,num_workers= num_workers)
        model = model_sel(model_name)
        
        ## model
        rl2model = model
        total_params  = sum(p.numel() for p in rl2model.parameters() if p.requires_grad)
        print("Trainable parameters : " + str(total_params))
        rl2model = rl2model.to(device)

        rl2model = torch.nn.DataParallel(rl2model)
        torch.backends.cudnn.benchmark = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, rl2model.parameters()) ,lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.50)
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        history = []


        for epoch in range(num_epochs):

            train_loss, train_acc=train_epoch(rl2model, train_dataloader, criterion, optimizer, epoch)
            valid_loss, valid_acc, val_roc_auc = valid_epoch(rl2model, val_dataloader, criterion)
            scheduler.step() 

            val_lr = optimizer.param_groups[0]['lr']

            # Print training and validation results
            print( f'\nEpoch: {epoch +1} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}' )
            print( f'\t\tTraining Accuracy: { train_acc:.2f}%\t Validation Accuracy: {valid_acc:.2f}%' )
            print(f'\t\tAUC: {val_roc_auc:.4f}, LR: {val_lr}')

            #save
            history.append([train_loss, valid_loss, train_acc, valid_acc])
            fmodel = f"fold_{fold+1}_epoch_{epoch+1:01d}-val_loss_{valid_loss:.4f}_val_accuracy_{valid_acc:.4f}_AUC_{val_roc_auc:.4f}.pth"
            torch.save(rl2model.state_dict(), os.path.join(otpath,fmodel))


        with open(os.path.join(otpath,f"Fold_{fold+1}_history.pickle"),'wb' ) as f:
            pickle.dump(history, f)



## Data
CRCtrain = pd.read_csv("CRCtrain.txt"),sep= "\t")
STADtrain = pd.read_csv("STADtrain"),sep= "\t")
UCECtrain = pd.read_csv("UCECtrain.txt",sep= "\t")
PANtrain = pd.read_csv("PANtrain.txt"),sep= "\t")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## RUNNING
for model_name in modelList:
    prj ="crc"
    otpath = model_name+"_"+prj
    run_train(CRCtrain, model_name, device, otpath)

modelList=["efficientnet", "resnet18","vgg19"]
for model_name in modelList:
    prj ="stad"
    otpath = model_name+"_"+prj
    run_train(STADtrain, model_name, device, otpath)


for model_name in modelList:
    prj ="ucec"
    otpath = model_name+"_"+prj
    run_train(UCECtrain, model_name, device, otpath)


for model_name in modelList:
    prj ="pan"
    otpath = model_name+"_"+prj
    run_train(PANtrain, model_name, device, otpath)


