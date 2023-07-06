# Author: Shilong
# Creat Data: 2023-6
# Modify Date:2023-7.6
# Description: realize amounts of functions as below:
# 1. 
# 2.

import os 
import shutil
import re
import wave
import librosa
import matplotlib.pyplot as plt 
import librosa.display
import soundfile
import csv
import pyaudio
import pylab
import numpy as np
import pandas as pd
import scipy
# import wfdb
from pydub import AudioSegment

# make dictionary
def mkdir(path):
    folder = os.path.exists(path)
    # judge wether make dir or not
    if not folder:                   
        os.makedirs(path)  

# read csv file by column
# train_csv=pd.read_csv("F:/heart_data/2022_challenge/heart/training_data.csv",sep=',')
def csv_reader_cl(file_name,clo_num):
    with open(file_name,encoding='utf-8') as csvfile:
        reader=csv.reader(csvfile)
        column=[row[clo_num] for row in reader]
    return column

# read the csv row_num-th row
def csv_reader_row(file_name,row_num):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        row=list(reader)
    return row[row_num]


# copy wav file to folder_path
# E:\01_Work\02_SoftwareDesign\circor_digiscope_dataset\Absent\2530\2530_AV
def copy_file(src_path,folder_path,patient_id_list,mur,position):
    """将文件复制到目标目录"""    
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir=folder_path+"\\"+mur+"\\"+patient_id+"\\"
            os.makedirs(target_dir, exist_ok=True)
            
            txtname=src_path+"\\"+patient_id+".txt"
            wavname=src_path+"\\"+patient_id+pos+".wav"    
            heaname=src_path+"\\"+patient_id+pos+".hea"
            tsvname=src_path+"\\"+patient_id+pos+".tsv"

            if os.path.exists(txtname):
                shutil.copy(txtname, target_dir+"\\")
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir+"\\")
            if os.path.exists(heaname):
                shutil.copy(heaname, target_dir+"\\")
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir+"\\")

# copy wav file to folder_path
# E:\01_Work\02_SoftwareDesign\circor_digiscope_dataset\Absent\2530\2530_AV
def copy_wav_file(src_path,folder_path,patient_id_list,mur,position):
    """将文件复制到目标目录"""    
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir=folder_path+"\\"+mur+"\\"+patient_id+"\\"
            os.makedirs(target_dir, exist_ok=True)
            
            # txtname=src_path+"\\"+patient_id+".txt"
            wavname=src_path+"\\"+patient_id+pos+".wav"    
            # heaname=src_path+"\\"+patient_id+pos+".hea"
            tsvname=src_path+"\\"+patient_id+pos+".tsv"

            # if os.path.exists(txtname):
            #     shutil.copy(txtname, target_dir+"\\")
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir+"\\")
            # if os.path.exists(heaname):
            #     shutil.copy(heaname, target_dir+"\\")
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir+"\\")

def index_load(tsvname):
   #读取tsv文件内容,不需要close函数
   with open(tsvname, 'r') as f:
    txt_data = f.read()
    head=['start','end','period']
    data=txt_data.split('\n')[:-1]
    #遍历每一行
    for l in data:
        sgmt=l.split('\t')
        if sgmt[2]!='0':
            head=np.vstack([head,sgmt])
   return head[1:]

# devide sounds into 4s segments
def pos_dir_make(dir_path,patient_id,pos):
    for po in pos:
        subdir=dir_path+patient_id+'\\'+patient_id+po
        wavname=subdir+".wav"
        if os.path.exists(wavname):
            print("exist")
            mkdir(subdir)# make dir

# preprocessed PCGs were segmented into four heart sound states
def state_div(tsvname,wavname,state_path,index):
    # 获取周期时间点
    # type:array
    index_file=index_load(tsvname)
    # print(index_file)
    # 获取array长度：a.shape[0]
    # 分割音频
    # states_num=[0,0,0,0]
    num=0
    start_index=0
    end_index=0

    for i in range(index_file.shape[0]-3):
        if index_file[i][2]=='1'and index_file[i+3][2]=='4':
            start_index=float(index_file[i][0])*1000
            end_index=float(index_file[i+3][1])*1000
            num=num+1
        print(start_index,end_index)
        # print("wav index")
        # print(wav_index)
        
        # print(start_index,end_index)
        # period_index=wav_index[2]
        
        # states_index=switch_period(wav_index[2])
        recording, fs = librosa.load(wavname,sr=1000)
        print("================================================================")
        print("wav name: "+wavname)
        # print("fs is: "+str(fs))

        buff = recording[int(start_index) :  int(end_index) ]  # 字符串索引切割
        print("buff len: "+str(len(buff)))
        # print("buff : "+str(buff))
        # print(buff)
        # buff.export(folder_path+mur+patient_id+"\\"+patient_id+pos+"{}{}+.wav".format(period_index,i), format="wav")
        # cut.append(buff)
        soundfile.write(state_path+"{}_{}.wav".format(index,num),buff,fs)

# tsv=r'E:\Shilong\murmur\LM_wav_dataset\Absent\2530\2530_TV.tsv'
# wav=r'E:\Shilong\murmur\LM_wav_dataset\Absent\2530\2530_TV.wav'
# path="E:\\Shilong\\murmur\\LM_wav_dataset\\Absent\\2530\\"
# state_div(tsv,wav,path,'a')

csv_path="E:\\Shilong\\murmur\\dataset_all\\training_data.csv"
# csv_path="E:\\Shilong\\murmur\\circor_digiscope_dataset\\training_data.csv"

# get dataset tag from table
row_line=csv_reader_row(csv_path,0)
tag_list=list()
# get index for 'Patient ID' and 'Outcome'
tag_list.append(row_line.index('Patient ID'))
tag_list.append(row_line.index('Murmur'))
# for tag_index in tag_list:
id_data=csv_reader_cl(csv_path,tag_list[0])
outcome_data=csv_reader_cl(csv_path,tag_list[1])

# init aptient id list for absent present and unknown
absent_patient_id=list()
present_patient_id=list()
unknown_patient_id=list()
# get 'Absent' and 'Present' and 'Unknown' index
absent_id =[out for out,outcome_data in enumerate(outcome_data) if outcome_data=='Absent']
present_id=[out for out,outcome_data in enumerate(outcome_data) if outcome_data=='Present']
unknown_id=[out for out,outcome_data in enumerate(outcome_data) if outcome_data=='Unknown']
# get 'Absent' and 'Present' and 'Unknown' patients ID
for id in absent_id:
    absent_patient_id.append(id_data[id])
for id in present_id:
    present_patient_id.append(id_data[id])
for id in unknown_id:
    unknown_patient_id.append(id_data[id])


# save patient id as csv
pd.DataFrame(data = absent_patient_id,index = None).to_csv('absent_id.csv')
pd.DataFrame(data = present_patient_id,index = None).to_csv('present_id.csv')
pd.DataFrame(data = unknown_patient_id,index = None).to_csv('unknown_id.csv')

# digaiation position
# define path options
positoin=['_AV','_MV','_PV','_TV']
murmur=["Absent\\","Present\\"]
period=["s1", "systolic", "s2", "diastolic"]
folder_path=r'E:\Shilong\murmur\LM_wav_dataset\\'
patient_id=absent_patient_id[0]
mur=murmur[0]
pos=positoin[0]

# # make dir and copy wav_files for Absent/ Present/ Unknown parients
# src_path=r'E:\Shilong\murmur\dataset_all\training_data'
# folder_path=r'E:\Shilong\murmur\LM_wav_dataset\\'
# copy_file(src_path,folder_path,absent_patient_id,"Absent",positoin)
# # # make dir and copy files for Present parients
# copy_file(src_path,folder_path,present_patient_id,"Present",positoin)
# # # make dir and copy files for Unkonwn parients
# copy_file(src_path,folder_path,unknown_patient_id,"Unknown",positoin)


src_path=r'E:\Shilong\murmur\dataset_all\training_data'

copy_wav_file(src_path,folder_path,absent_patient_id,"Absent",positoin)
# # make dir and copy files for Present parients
copy_wav_file(src_path,folder_path,present_patient_id,"Present",positoin)
# # make dir and copy files for Unkonwn parients
copy_wav_file(src_path,folder_path,unknown_patient_id,"Unknown",positoin)

# make dir for each position
# E:\Shilong\murmur\LM_wav_dataset
src_path=r'E:\Shilong\murmur\dataset_all\training_data'
folder_path=r'E:\Shilong\murmur\LM_wav_dataset\\'
for mur in murmur:
    dir_path=folder_path+mur
    for patient_id in absent_patient_id:
        pos_dir_make(dir_path,patient_id,positoin)
    for patient_id in present_patient_id:
        pos_dir_make(dir_path,patient_id,positoin)
    for patient_id in unknown_patient_id:
        pos_dir_make(dir_path,patient_id,positoin)


# 切数据，命名格式为：id+pos+state+num
file_path=folder_path+mur+patient_id+patient_id+pos+".tsv"
# path="E:\\01_Work\\02_SoftwareDesign\\circor_dataset_period\\"
def period_div(path,murmur,patient_id_list,positoin,tg):
    for mur in murmur:
        for patient_id in patient_id_list:
            for pos in positoin:
                dir_path=path+mur+patient_id+"\\"+patient_id+pos
                tsv_path=dir_path+".tsv"
                wav_path=dir_path+".wav"
                if os.path.exists(tsv_path):
                    state_div(tsv_path,wav_path,dir_path+"\\",patient_id+pos)

period_div(folder_path,murmur,absent_patient_id,positoin,'a_')
period_div(folder_path,murmur,present_patient_id,positoin,'p_')
# period_div(folder_path,murmur,unknown_patient_id,positoin)

# 保存absent和present id并读入为list
# read csv
import csv
absent_csv=r'E:\Shilong\murmur\03_Classifier\MurmurDectection\absent_id.csv'
present_csv=r'E:\Shilong\murmur\03_Classifier\MurmurDectection\present_id.csv'
# absent_csv=r'E:\Shilong\murmur\03_Classifier\MurmurDectection\uknown_id.csv'
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row['0'] for row in reader]   # weight 同列的数据
        return id
absent_patient_id=get_patientid(absent_csv)
present_patient_id=get_patientid(present_csv)

import random
# 将absent_id和present_id按照7：3随机选取id划分为训练集和测试集
absent_train_id=random.sample(absent_patient_id,int(len(absent_patient_id)*0.7))
present_train_id=random.sample(present_patient_id,int(len(present_patient_id)*0.7))
absent_test_id=list(set(absent_patient_id)-set(absent_train_id))
present_test_id=list(set(present_patient_id)-set(present_train_id))

import shutil,os
# 将训练集和测试集文件分别copy到train和test文件夹
positoin=['_AV','_MV','_PV','_TV']
murmur=["Absent\\","Present\\"]
period=["s1", "systolic", "s2", "diastolic"]

folder=r'E:\Shilong\murmur\LM_wav_dataset'

for patient_id in absent_train_id:    
    dir_path= folder+"\\absent\\"+patient_id
    print(dir_path)
    for root,dir,file in os.walk(dir_path):
        for subdir in dir:
            subdir_path=os.path.join(root,subdir)
            print(subdir_path)
            # if os.path.exists(dir_path):
            shutil.copytree(subdir_path,folder+"\\train\\absent\\"+subdir)