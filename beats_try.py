# Author: Shilong
# Creat Data: 2023-6-16
# Modify Date:2023-
# Description: load fin-tuned BEATs model  

"""
Load Pre-Trained Models
"""
import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil,os
import random
import csv
import librosa
import time
import logging
from datetime import datetime
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
torch.__version__
from BEATs import BEATs, BEATsConfig,BEATs_Pre_Train_itere3
from BEATs_def import get_patientid,get_mfcc_features,copy_wav,get_mel_features,csv_reader_cl

# ========================/ parameteres define /========================== # 
murmur_positoin=['_AV','_MV','_PV','_TV']
murmur_ap=["Absent\\","Present\\"]
period=["s1", "systolic", "s2", "diastolic"]

# file_path=r'E:\Shilong\murmur\circor_dataset_period\train'
# get absent / present patient_id
# absent_csv=r'E:\Shilong\murmur\03_Classifier\MurmurDectection\absent_id.csv'
# present_csv=r'E:\Shilong\murmur\03_Classifier\MurmurDectection\present_id.csv'
# absent_patient_id=get_patientid(absent_csv)
# present_patient_id=get_patientid(present_csv)

# ========================/ load model /========================== # 
# load the pre-trained checkpoints
# checkpoint = torch.load(r'E:\Shilong\murmur\03_Classifier\LM\LM_Model\BEATs\BEATs_iter3.pt')

# cfg = BEATsConfig(checkpoint['cfg'])
# BEATs_model = BEATs(cfg)
# BEATs_model.load_state_dict(checkpoint['model'])
# # BEATs_model.eval()
# # extract the the audio representation
# # audio_input_16khz = torch.randn(2, 10000)

# probs = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
# representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

# ========================/ model define /========================== # 
MyModel=BEATs_Pre_Train_itere3()

# ========================/ dataset Class /========================== #
class MyDataset(Dataset): 
    """ my dataset."""    
    # Initialize your data, download, etc.
    def __init__(self,wavlabel,wavdata):
        # 直接传递data和label
        # self.len = wavlen
        self.data = torch.from_numpy(wavdata)
        self.label = torch.from_numpy(wavlabel)
               
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = torch.Tensor(self.data[index])
        labelitem = torch.Tensor(self.label[index])
        return dataitem.float(), labelitem.float()
        
    def __len__(self): 
        # 返回文件数据的数目
        return len(self.data)
    
# ========================/ logging init /========================== #
def logger_init(log_level=logging.DEBUG,
                log_dir=r'E:\Shilong\murmur\03_Classifier\LM\BEATs\ResultFile',
                ):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    date=datetime.now()
    log_path = os.path.join(log_dir,str(date)[:13] +'-'+str(date.minute)+ '.log')
    formatter = '[%(asctime)s - %(levelname)s:] %(message)s'
    logging.basicConfig(level=log_level,
                        format=formatter,
                        datefmt='%Y-%d-%m %H:%M:%S',
                        handlers=[logging.FileHandler(log_path),
                                logging.StreamHandler(sys.stdout)]
                            )

# ========================/ logging formate /========================== #
class save_info(object):
    def __init__(self,epoch,train_loss,test_acc,test_loss) :
        self.epoch=epoch
        self.train_loss=train_loss
        self.test_acc=test_acc
        self.test_loss=test_loss
        logging.info(f"epoch: "+str(self.epoch))
        logging.info(f"train_loss: "+str('{:.3f}'.format(self.train_loss)))
        logging.info(f"test_acc: "+str('{:.3f}%'.format(self.test_acc))+"  test_loss: "+str('{:.3f}'.format(self.test_loss)))
        logging.info(f"=========================================")

# ========================/ file path /========================== # 
absent_train_csv_path=r'E:\Shilong\murmur\LM_wav_dataset\train_csv'
absent_test_csv_path=r'E:\Shilong\murmur\LM_wav_dataset\test_csv'
present_train_csv_path=r'E:\Shilong\murmur\LM_wav_dataset\train_csv'
present_test_csv_path=r'E:\Shilong\murmur\LM_wav_dataset\test_csv'

filepath=r'E:\Shilong\murmur\circor_dataset_period'
absent_train_path=r'E:\Shilong\murmur\LM_wav_dataset\train\Absent'
absent_test_path=r'E:\Shilong\murmur\LM_wav_dataset\test\Absent'
Present_train_path=r'E:\Shilong\murmur\LM_wav_dataset\train\Present'
present_test_path=r'E:\Shilong\murmur\LM_wav_dataset\test\Present'



folder=r'E:\Shilong\murmur\LM_wav_dataset'
npy_path=r'E:\Shilong\murmur\LM_wav_dataset\npyFile'

# ========================/ devide trainset and testset /========================== #

"""# 将absent_id和present_id按照7:3随机选取id划分为训练集和测试集
absent_train_id=random.sample(absent_patient_id,int(len(absent_patient_id)*0.7))
present_train_id=random.sample(present_patient_id,int(len(present_patient_id)*0.7))
absent_test_id=list(set(absent_patient_id)-set(absent_train_id))
present_test_id=list(set(present_patient_id)-set(present_train_id))



# 读取csv文件
absent_train_id = csv_reader_cl(absent_train_id_path,0)
present_train_id =csv_reader_cl(present_train_id_path,0)
absent_test_id = csv_reader_cl(absent_test_id_path,0)
present_test_id = csv_reader_cl(present_test_id_path,0)

# 将wav文件拷到对应的文件夹中
copy_wav(folder,absent_test_id,'Absent',absent_test_path)
copy_wav(folder,present_test_id,'Present',present_test_path)
copy_wav(folder,absent_train_id,'Absent',absent_train_path)
copy_wav(folder,present_train_id,'Present',Present_train_path)
"""

positoin=['_AV','_MV','_PV','_TV']
murmur=["Absent\\","Present\\"]
period=["s1", "systolic", "s2", "diastolic"]

# ========================/ get wav data, length=10000 /========================== # 
# absent_train_features=get_mfcc_features(BEATs_model,absent_train_path,absent_train_csv_path,padding_mask)# absent
# absent_test_features=get_mfcc_features(BEATs_model,absent_test_path,absent_test_csv_path,padding_mask)# absent
# present_train_features=get_mfcc_features(BEATs_model,Present_train_path,present_train_csv_path,padding_mask)# present
# present_test_features=get_mfcc_features(BEATs_model,present_test_path,present_test_csv_path,padding_mask)# present

# # ========================/ save as npy file /========================== # 
# np.save(npy_path+r'\absent_train_features.npy',absent_train_features)
# np.save(npy_path+r'\absent_test_features.npy',absent_test_features)
# np.save(npy_path+r'\present_train_features.npy',present_train_features)
# np.save(npy_path+r'\present_test_features.npy',present_test_features)

# ========================/ load npy file /========================== # 
absent_train_features = np.load(npy_path+r'\absent_train_features.npy',allow_pickle=True)
absent_test_features = np.load(npy_path+r'\absent_test_features.npy',allow_pickle=True)
present_train_features = np.load(npy_path+r'\present_train_features.npy',allow_pickle=True)
present_test_features = np.load(npy_path+r'\present_test_features.npy',allow_pickle=True)

# ========================/ get features & labels /========================== # 
path=r'E:\Shilong\murmur\LM_wav_dataset\csv'
train_path=r'E:\Shilong\murmur\LM_wav_dataset\train_csv'
test_path=r'E:\Shilong\murmur\LM_wav_dataset\test_csv'
# test_features,test_label=get_mel_features(path,absent_patient_id,present_patient_id)

"""
train_features,train_label=get_mel_features(train_path,absent_patient_id,present_patient_id)
test_features,test_label=get_mel_features(test_path,absent_patient_id,present_patient_id)
train_features=train_features.astype(float)
train_label=train_label.astype(float)
test_features=test_features.astype(float)
test_label=test_label.astype(float)
"""
# ========================/ label encoder /========================== # 
absent_train_label=np.ones(absent_train_features.shape[0])
absent_test_label=np.ones(absent_test_features.shape[0])
present_train_label=np.zeros(present_train_features.shape[0])
present_test_label=np.zeros(present_test_features.shape[0])

train_label=np.hstack((absent_train_label,present_train_label))
test_label=np.hstack((absent_test_label,present_test_label))
train_features=np.vstack((absent_train_features,present_train_features))
test_features=np.vstack((absent_test_features,present_test_features))

# ========================/ train test /========================== # 
train_features=train_features.astype(float)
train_label=train_label.astype(float)
test_features=test_features.astype(float)
test_label=test_label.astype(float)

# ========================/ MyDataset /========================== # 
train_set=MyDataset(wavlabel=train_label,wavdata=train_features)
test_set=MyDataset(wavlabel=test_label,wavdata=test_features)

# ========================/ HyperParameters /========================== # 
train_batch_size= 128
test_batch_size = 128
learning_rate = 0.001
num_epochs = 50
padding_size = 500
padding = torch.zeros(train_batch_size, padding_size).bool() # we randomly mask 75% of the input patches,
padding_mask=torch.Tensor(padding)

# ========================/ dataloader /========================== # 
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True,drop_last=True)
print("Dataloader is ok") # 最后再打印一下新的模型

# ========================/ model add fc-layer /========================== # 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MyModel.to(DEVICE)# 放到设备中
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params':MyModel.last_layer.parameters()}
], lr=learning_rate)#指定 新加的fc层的学习率

# ========================/ train model /========================== # 
# 定义训练函数

def train_model(model,device, train_loader, test_loader,padding,epoch):
# train model
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        padding=padding.to(device)
        optimizer.zero_grad()
        y_hat= model(x,padding)
        loss = criterion(y_hat, y.long())
        loss.backward()
        optimizer.step()

# evaluate model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):          
            x,y= data
            x=x.to(device)
            y=y.to(device)
            padding=padding.to(device)
            optimizer.zero_grad()
            y_hat = model(x,padding)
            test_loss += criterion(y_hat, y.long()).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    writer.add_scalar("train_loss",loss,epoch)
    writer.add_scalar("test_loss",test_loss,epoch)
    writer.add_scalar("test_acc",100. * correct / len(test_set),epoch)
    save=save_info(epoch,loss.item(),100. * correct / len(test_set),test_loss)

# ========================/ training and logging info /========================== # 

logger_init()
logging.info("# train_batch_size = "+str(train_batch_size))
logging.info("# test_batch_size = "+str(test_batch_size))
logging.info("# learning_rate = "+str(learning_rate))
logging.info("# num_epochs = "+str(num_epochs))
logging.info("# padding_size = "+str(padding_size))
logging.info("----------------------------------------------------")
writer = SummaryWriter(r'./tensorboard/'+str(datetime.now())[:13])
for epoch in range(num_epochs):
    train_model(model=MyModel,device=DEVICE, train_loader=train_loader,test_loader=test_loader,padding=padding_mask,epoch=epoch)

writer.close()


