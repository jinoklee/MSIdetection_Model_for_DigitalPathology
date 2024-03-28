import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from PIL import Image
import cv2



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True


class CustomImageDataset_from_csv(Dataset):
    def __init__(self, dataframe ,   transform = None):
        self.img_labels = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self , idx):
        img_path =  self.img_labels.iloc[idx, 5]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 6]
        if self.transform:
            image = self.transform(image)
        return(image, label)

#### image  Transfomation ####
image_transform = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(5/32, 5/32), shear = 0),
        transforms.RandomRotation(degrees=25),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1).reshape(-1,1)

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

def inference(data_loader, model, device):
    random_state = set_seed(42)
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, position=0, leave=True, desc='Inference'):
            inputs = data.to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy().tolist()
            predictions.extend(outputs)

    return predictions



def msi(data ,CFG, device, modelpt, ver):

    Batch_Size = CFG['batch_size']
    num_classes = CFG['num_classes']
    model_dir = CFG['model_dir']
    model_name = CFG['baseline']

    if ver == "V1":
        model = model_selV1(model_name)

    if ver == "V2":
        model = model_selV2(model_name)

    if ver == "V3":
        model = model_selV3(model_name)

    print('MODEL LOADED :'+model_name)

    df_result = {}
    df_result['path'] =data['path'].values

    df_result['total'] =[]
    PATH = os.path.join(model_dir, modelpt)

    model.to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    dataset = CustomImageDataset_from_csv(data, transform = image_transform['test'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, shuffle=False, num_workers=22)
    predictions = inference(data_loader=loader, model=model, device=device)

    df_result['total'].extend(softmax(np.asarray(predictions))[:,1]) #MSI sore

    df_result = pd.DataFrame(df_result)
    return df_result



def auc_cross(df,test_df):
    test_class=test_df.iloc[:,[5,6]]
    df = df.merge(test_class, on = "path", how = 'left')
    
    fname = []
    label = []
    for i in range(len(df)):
        slide = df.iloc[i,1].split("/")[-1]
        slide = slide[0:12]
        slabel = df.iloc[i,3]
        fname.append(slide)
        label.append(slabel)

    df['fname'] = fname
    df['label'] = label
    fname = list(set(df['fname']))

    total_score = []
    total_label = []
    for i in fname:
        isDF = df.loc[df['fname']==i]
        score = sum(isDF['total'])/isDF.shape[0]
        label = ''.join(set(isDF['label']))
        total_score.append(score)
        total_label.append(label)

    total = pd.DataFrame({'fname':fname,
                         'score': total_score,
                         'label' : total_label})
    
    mss_score = total.loc[total['label']=='MSS']
    msi_score = total.loc[total['label']=='MSI-H']


    threshold = []
    gap = (max(total_score)- min(total_score))/(len(total_score)-1)

    for i in range(len(total_score)):
        threshold.append(max(total_score)-gap*i)
        
        
    tpr_roc_1 = []
    for i in range(len(total_score)):
        thresh = threshold[i]
        pred_count = 0
        for i in list(mss_score['score'].values) :
            if i >= thresh: pred_count+=1
        tpr_roc_1.append(pred_count/len(list(mss_score['score'].values)))

    fpr_roc_1 = []
    for i in range(len(total_score)):
        thresh = threshold[i]
        pred_count = 0
        for i in msi_score['score'].values:
            if i>=thresh: pred_count+=1
        fpr_roc_1.append(pred_count/len(msi_score['score'].values))

    threshold = np.array(threshold)
    fpr_roc = np.array(fpr_roc_1)
    tpr_roc = np.array(tpr_roc_1)
    auc_roc = auc(fpr_roc_1, tpr_roc_1)
    
    return threshold, fpr_roc, tpr_roc, auc_roc

