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





## Data
CRCtrain = pd.read_csv("CRCtrain.txt"),sep= "\t")
CRCtest = pd.read_csv("CRCtest.txt"),sep= "\t")

STADtrain = pd.read_csv("STADtrain"),sep= "\t")
STADtest = pd.read_csv("STADtest.txt"),sep= "\t")

UCECtrain = pd.read_csv("UCECtrain.txt",sep= "\t")
UCECtest = pd.read_csv("UCECtest.txt",sep= "\t")

PANtrain = pd.read_csv("PANtrain.txt"),sep= "\t")
PANtest = pd.read_csv("PANtest.txt"),sep= "\t")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

## RUNNING
for model_name in modelList:
    prj ="crc"
    otpath = "/BiO/jolee/02.TCGA/ModelTest/06.model/20231215_"+model_name+"_"+prj
    run_train(CRCtest, CRCtrain, model_name, device, otpath)

modelList=["efficientnet", "resnet18","vgg19"]
for model_name in modelList:
    prj ="stad"
    otpath = "/BiO/jolee/02.TCGA/ModelTest/06.model/20231215_"+model_name+"_"+prj
    run_train(STADtest, STADtrain, model_name, device, otpath)


for model_name in modelList:
    prj ="ucec"
    otpath = "/BiO/jolee/02.TCGA/ModelTest/06.model/20231215_"+model_name+"_"+prj
    run_train(UCECtest, UCECtrain, model_name, device, otpath)


for model_name in modelList:
    prj ="pan"
    otpath = "/BiO/jolee/02.TCGA/ModelTest/06.model/20231215_"+model_name+"_"+prj
    run_train(PANtest, PANtrain, model_name, device, otpath)


