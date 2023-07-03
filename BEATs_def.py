import csv
import os
import librosa
import torch,shutil
import pandas as pd
import numpy as np

def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row['0'] for row in reader]   # weight 同列的数据
    return id


def get_mfcc_features(BEATs_model,dir_path,csv_path,padding_mask):    
    for root,dir,file in os.walk(dir_path):
        for subfile in file:
            wav_path=os.path.join(root,subfile)
            
            if os.path.exists(wav_path):
                print("reading: "+subfile)
                y, sr = librosa.load(wav_path, sr=2000)
                y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                wav = torch.tensor(y_16k)
                # 增加一个维度满足输入要求
                # wav = wav.unsqueeze(0)
                # padding=torch.zeros(16-wav.shape[0], wav.shape[1]).bool()
                rep = BEATs_model.extract_features(wav)[0]
                # rep1 = BEATs_model.extract_features(wav)[0]
                rep = torch.squeeze(rep).detach().cpu().numpy()
                feature = pd.DataFrame(rep)
                save_path = csv_path +"\\"+ subfile+ ".csv"
                # print("shape: "+feature.shape())
                feature.to_csv(save_path, index=False, header=False)
    

def copy_wav(folder,idlist,mur,traintest):
    for patient_id in idlist:    
        dir_path= folder+"\\"+mur+"\\"+patient_id
        # print(dir_path)
        for root,dir,file in os.walk(dir_path):
            for subdir in dir:
                subdir_path=os.path.join(root,subdir)
                print(subdir_path)
                # if os.path.exists(dir_path):
                shutil.copytree(subdir_path,traintest+"\\"+subdir)
    # idlist.to_scv(folder+"\\"+traintest+"\\"+mur+traintest+".csv", index=False, header=False)

"""
读取csv文件返回feature和label
"""
def get_mel_features(dir_path,absent_id,present_id):
    feature_list=[]
    label_list=[]
    for root,dir,file in os.walk(dir_path):
        for subfile in file:
            csv_path=os.path.join(root,subfile)
            print("reading: "+subfile)
            # filename = r'E:\Shilong\murmur\LM_wav_dataset\train_csv\2530_AV_1.wav.csv'
            df = pd.read_csv(csv_path,header=None)
            data = np.array(df)
            if data.shape[0]<24:
                data = np.pad(data,(0,24-data.shape[0]),'constant',constant_values=(0,0))
                data = data[:,0:768]
            elif  data.shape[0]>24:
                data=data[0:24,:]
            feature_list.append(data)
            id=subfile.split('_')[0]
            if id in absent_id:
                label_list.append(1)
            if id in present_id:
                label_list.append(0)
        
    return np.array(feature_list),np.array(label_list)