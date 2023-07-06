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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
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
checkpoint = torch.load(r'E:\Shilong\murmur\03_Classifier\LM\LM_Model\BEATs\BEATs_iter3.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
# BEATs_model.eval()
# extract the the audio representation
# audio_input_16khz = torch.randn(2, 10000)
padding_mask = torch.zeros(1, 800).bool()
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

absent_train_id_path = r'E:\Shilong\murmur\LM_wav_dataset\absent_train_id_path.csv'
absent_test_id_path = r'E:\Shilong\murmur\LM_wav_dataset\absent_test_id_path.csv'
present_train_id_path = r'E:\Shilong\murmur\LM_wav_dataset\present_train_id_path.csv'
present_test_id_path = r'E:\Shilong\murmur\LM_wav_dataset\present_test_id_path.csv'

folder=r'E:\Shilong\murmur\LM_wav_dataset'
npy_path=r'E:\Shilong\murmur\LM_wav_dataset\npyFile'
# ========================/ devide trainset and testset /========================== #

"""# 将absent_id和present_id按照7:3随机选取id划分为训练集和测试集
absent_train_id=random.sample(absent_patient_id,int(len(absent_patient_id)*0.7))
present_train_id=random.sample(present_patient_id,int(len(present_patient_id)*0.7))
absent_test_id=list(set(absent_patient_id)-set(absent_train_id))
present_test_id=list(set(present_patient_id)-set(present_train_id))

# 保存train、test id为CSV文件
absent_train_id=pd.DataFrame(absent_train_id)
present_train_id=pd.DataFrame(present_train_id)
absent_test_id=pd.DataFrame(absent_test_id)
present_test_id=pd.DataFrame(present_test_id)

absent_train_id.to_csv(absent_train_id_path, index=False, header=False)
present_train_id.to_csv(present_train_id_path, index=False, header=False)
absent_test_id.to_csv(absent_test_id_path, index=False, header=False)
present_test_id.to_csv(present_test_id_path, index=False, header=False)

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
train_batch_size= 64
test_batch_size = 64
# learning_rate = 0.001
num_epochs = 80

# ========================/ dataloader /========================== # 
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# save_features(patient_id_list=absent_patient_id)
# save_features(patient_id_list=present_patient_id)
# ========================/ fine-tuning /========================== #


# ========================/ model add fc-layer /========================== # 

# 将所有的参数层进行冻结
# for param in MyModel.parameters():
#     param.requires_grad = False  # 参数层进行冻结
# # 这里打印下全连接层的信息
# print("MyModel.last_layer")
# print(MyModel.last_layer)
# num_fc_ftr = MyModel.last_layer.in_features #获取到fc层的输入
# MyModel.fc = nn.Linear(num_fc_ftr, 2) # 定义一个新的FC层


model=MyModel.to(DEVICE)# 放到设备中
print(model) # 最后再打印一下新的模型
# ========================/ train parameters /========================== # 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params':MyModel.last_layer.parameters()}
], lr=0.001)#指定 新加的fc层的学习率

# 定义训练函数
def train(model,device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat= model(x)
        loss = criterion(y_hat, y.long())
        loss.backward()
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):          
            x,y= data
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y.long()).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_set),
            100. * correct / len(test_set)))
# ========================/ training model /========================== # 
# 训练epochs=9
for epoch in range(1, 10):
    train(model=MyModel,device=DEVICE, train_loader=test_loader,epoch=epoch)
    test(model=MyModel, device=DEVICE, test_loader=test_loader)
    train_log_filename = "train_log.txt"
    result_dir=r'E:\Shilong\murmur\03_Classifier\LM\logs'
    train_log_filepath = os.path.join(result_dir, train_log_filename)
    train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    to_write=train_log_txt_formatter.format(time_str=time.strtime("%Y_%n_%d_%H:%M:%S"),epoch=epoch,loss_str=" ".join(["{}".format(loss)]))
    with open(train_log_filepath,"a") as f:
        f.write(to_write)

