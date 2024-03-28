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


def run(ver, test_df, tesetprj, device, modelpt):

    le = preprocessing.LabelEncoder()
    test_df['status'] = le.fit_transform(test_df['Status'])
    CFG = {'fold':1, 
           'baseline': model_name,
           'freeze_layer':None,
           'batch_size':256,
           'num_classes':2,
           'model_dir':otpath}

    df_result = msi(data = test_df,
                    CFG = CFG,
                    device = device,
                    modelpt = modelpt,
                    ver=ver)

    df_result.to_csv( os.path.join(otpath,testprj+"_whole.txt"), sep="\t")
    df_1 = pd.read_csv(os.path.join(otpath,testprj+"_whole.txt"),sep= "\t")
    threshold_1, fpr_roc_1, tpr_roc_1, auc_roc_1 = auc_cross(df_1,test_df)

    plt.figure(figsize=(3, 3))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_roc_1, tpr_roc_1, label='AUC (area = {:.3f})'.format(auc_roc_1))
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(otpath,testprj+"_auc.png"))
    print("PATH"+otpath)
    print("model"+model_name+":"+prj)
    print("TestData"+testprj)
    print("Save done")



device = 'cuda' if torch.cuda.is_available() else 'cpu'

CRCtest = pd.read_csv("CRCtest",sep= "\t")
STADtest = pd.read_csv("STADtest.txt"),sep= "\t")
UCECtest = pd.read_csv("UCECtest.txt"),sep= "\t")

run(ver, CRCtest, "CRC", device)
run(ver, STADtest, "STAD", device)
run(ver, UCECtest, "UCEC", device)


