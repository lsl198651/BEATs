# Author: Shilong
# Creat Data: 2023-6
# Modify Date:2023-7.6
# Description: realize amounts of functions as below:
# 1. 
# 2.

import os 
import shutil
import random
import librosa
import matplotlib.pyplot as plt 
import librosa.display
import soundfile
import csv
import numpy as np
import pandas as pd
# import wfdb
from pydub import AudioSegment
# ========================/ functions define /========================== # 
# make dictionary
def mkdir(path):
    folder = os.path.exists(path)
    # judge wether make dir or not
    if not folder:                   
        os.makedirs(path)  

# read csv file by column
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
def copy_file(src_path,folder_path,patient_id_list,mur,position):
    """将所有文件复制到目标目录"""    
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
def copy_wav_file(src_path,folder_path,patient_id_list,mur,position):
    """将指定文件复制到目标目录"""
    # 1. make dir
    mur_dir=folder_path+"\\"+mur
    if not os.path.exists(mur_dir):
        os.makedirs(mur_dir)
    # 2. copy file
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir=folder_path+"\\"+mur+"\\"+patient_id+"\\"
            os.makedirs(target_dir, exist_ok=True)

            wavname=src_path+"\\"+patient_id+pos+".wav"
            tsvname=src_path+"\\"+patient_id+pos+".tsv"

            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir+"\\")
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir+"\\")

# devide sounds into 4s segments
def pos_dir_make(dir_path,patient_id,pos):
    for po in pos:
        subdir=dir_path+patient_id+'\\'+patient_id+po
        wavname=subdir+".wav"
        if os.path.exists(wavname):
            print("exist")
            mkdir(subdir)# make dir

def index_load(tsvname):
    """读取tsv文件内容,不需要close函数"""
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

# preprocessed PCGs were segmented into four heart sound states
def period_div(path,murmur,patient_id_list,positoin,id_data,Murmur_locations,Systolic_murmur_timing,Diastolic_murmur_timing):
    for mur in murmur:
        for patient_id in patient_id_list:
            for pos in positoin:
                dir_path=path+mur+patient_id+"\\"+patient_id+pos
                tsv_path=dir_path+".tsv"
                wav_path=dir_path+".wav"

                index = id_data.index(patient_id)
                wav_location = pos[1:]    # 听诊区域
                locations = Murmur_locations[index].split('+')  # 有杂音的区域
                # 此听诊区有杂音
                if wav_location in locations:
                    Systolic_state=Systolic_murmur_timing[index]
                    Diastolic_state=Diastolic_murmur_timing[index]
                    # 没有 Systolic murmur
                    if Systolic_state =='nan':
                        Systolic_murmur='Absent'
                    else: 
                        Systolic_murmur='Present'
                    # 没有 Diastolic murmur
                    if Diastolic_state =='nan':
                        Diastolic_murmur='Absent'
                    else: 
                        Diastolic_murmur='Present'

                # 此听诊区没有杂音
                else: 
                    Systolic_murmur='Absent'
                    Diastolic_murmur='Absent'
                    Systolic_state='nan'
                    Diastolic_state='nan'
                if os.path.exists(tsv_path):
                    state_div(tsv_path,wav_path,dir_path+"\\",patient_id+pos,Systolic_murmur,Diastolic_murmur,Systolic_state,Diastolic_state)

# preprocessed PCGs were segmented into four heart sound states
def state_div(tsvname,wavname,state_path,index,Systolic_murmur,Diastolic_murmur,Systolic_state,Diastolic_state):
    index_file=index_load(tsvname)
    recording, fs = librosa.load(wavname,sr=4000)
    num=0
    start_index2=0
    end_index2=0
    start_index4=0
    end_index4=0

    for i in range(index_file.shape[0]-3):
        if index_file[i][2]=='2'and index_file[i+2][2]=='4':
            start_index2=float(index_file[i][0])*fs
            end_index2=float(index_file[i][1])*fs
            start_index4=float(index_file[i+2][0])*fs
            end_index4=float(index_file[i+2][1])*fs
            num=num+1
            #  解决出现_0.wav的问题
            print(start_index2,end_index2,start_index4,end_index4)            
            print("=============================================")
            print("wav name: "+wavname)        
            buff2 = recording[int(start_index2) :int(end_index2) ]  # 字符串索引切割
            buff4 = recording[int(start_index4) :int(end_index4) ]  # 字符串索引切割
            print("buff2 len: "+str(len(buff2)),"buff4 len: "+str(len(buff4)))
            soundfile.write(state_path+"{}_{}_{}_{}_{}.wav".format(index,'Systolic' ,num,Systolic_murmur,Systolic_state),buff2,fs)
            soundfile.write(state_path+"{}_{}_{}_{}_{}.wav".format(index,'Diastolic',num,Diastolic_murmur,Diastolic_state),buff4,fs)


# get patient id from csv file
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row['0'] for row in reader]   # weight 同列的数据
        return id

# copy absent data to folder
def copy_states_data(folder,patient_id,murmur,type):
    din_path=folder+type+murmur
    if os.path.exists(din_path):
        os.makedirs(din_path)
    for id in patient_id:    
        dir_path= folder+murmur+id
        print(dir_path)
        for root,dir,file in os.walk(dir_path):
            for subdir in dir:
                subdir_path=os.path.join(root,subdir)
                print(subdir_path)
                # if os.path.exists(dir_path):
                shutil.copytree(subdir_path,din_path+subdir)

# ========================/ code executive /========================== # 
# ========================/ code executive /========================== # 
# ========================/ code executive /========================== # 

csv_path=r'E:\Shilong\murmur\dataset_all\training_data.csv'
# get dataset tag from table
row_line=csv_reader_row(csv_path,0)
tag_list=list()

# get index for 'Patient ID' and 'Outcome'
tag_list.append(row_line.index('Patient ID'))
tag_list.append(row_line.index('Murmur'))
tag_list.append(row_line.index('Murmur locations'))
tag_list.append(row_line.index('Systolic murmur timing'))
tag_list.append(row_line.index('Diastolic murmur timing'))

# for tag_index in tag_list:
id_data=csv_reader_cl(csv_path,tag_list[0])
Murmur=csv_reader_cl(csv_path,tag_list[1])
Murmur_locations=csv_reader_cl(csv_path,tag_list[2])
Systolic_murmur_timing=csv_reader_cl(csv_path,tag_list[3])
Diastolic_murmur_timing=csv_reader_cl(csv_path,tag_list[4])

# save data to csv file
Murmur_locations_path = r'E:\Shilong\murmur\03_circor_states\Murmur_locations.csv'
Systolic_murmur_timing_path = r'E:\Shilong\murmur\03_circor_states\Systolic_murmur_timing.csv'
Diastolic_murmur_timing_path = r'E:\Shilong\murmur\03_circor_states\Diastolic_murmur_timing.csv'

pd.DataFrame(Murmur_locations).to_csv(Murmur_locations_path, index=False, header=False)
pd.DataFrame(Systolic_murmur_timing).to_csv(Systolic_murmur_timing_path, index=False, header=False)
pd.DataFrame(Diastolic_murmur_timing).to_csv(Diastolic_murmur_timing_path, index=False, header=False)

# init aptient id list for absent present and unknown
absent_patient_id=list()
present_patient_id=list()

# get 'Absent' and 'Present' and 'Unknown' index
absent_id  = [out for out,Murmur in enumerate(Murmur) if Murmur=='Absent']
present_id = [out for out,Murmur in enumerate(Murmur) if Murmur=='Present']

# get 'Absent' and 'Present' and 'Unknown' patients ID
for id in absent_id:
    absent_patient_id.append(id_data[id])
for id in present_id:
    present_patient_id.append(id_data[id])

# save patient id as csv
# pd.DataFrame(data = absent_patient_id,index = None).to_csv('absent_id.csv', index=False, header=False)
# pd.DataFrame(data = present_patient_id,index = None).to_csv('present_id.csv', index=False, header=False)

# digaiation position
# define path options
positoin=['_AV','_MV','_PV','_TV']
murmur=["Absent\\","Present\\"]
period=["s1", "systolic", "s2", "diastolic"]
folder_path=r'E:\Shilong\murmur\03_circor_states\\'

src_path=r'E:\Shilong\murmur\dataset_all\training_data'
# # make dir and copy files for Present/Absent parients
copy_wav_file(src_path,folder_path,absent_patient_id,"Absent",positoin)
copy_wav_file(src_path,folder_path,present_patient_id,"Present",positoin)

# make dir for each position
# E:\Shilong\murmur\LM_wav_dataset
src_path=r'E:\Shilong\murmur\dataset_all\training_data'

for mur in murmur:
    dir_path=folder_path+mur
    for patient_id in absent_patient_id:
        pos_dir_make(dir_path,patient_id,positoin)
    for patient_id in present_patient_id:
        pos_dir_make(dir_path,patient_id,positoin)

# 切数据，命名格式为：id+pos+state+num
period_div(folder_path,murmur,absent_patient_id,positoin,id_data,Murmur_locations,Systolic_murmur_timing,Diastolic_murmur_timing)
period_div(folder_path,murmur,present_patient_id,positoin,id_data,Murmur_locations,Systolic_murmur_timing,Diastolic_murmur_timing)

absent_train_id_path = r'E:\Shilong\murmur\03_circor_states\absent_train_id.csv'
absent_test_id_path = r'E:\Shilong\murmur\03_circor_states\absent_test_id.csv'
present_train_id_path = r'E:\Shilong\murmur\03_circor_states\present_train_id.csv'
present_test_id_path = r'E:\Shilong\murmur\03_circor_states\present_test_id.csv'

# 将absent_id和present_id按照8:2随机选取id划分为训练集和测试集
absent_train_id=random.sample(absent_patient_id,int(len(absent_patient_id)*0.8))
present_train_id=random.sample(present_patient_id,int(len(present_patient_id)*0.8))
absent_test_id=list(set(absent_patient_id)-set(absent_train_id))
present_test_id=list(set(present_patient_id)-set(present_train_id))

# 将训练集和测试集文件分别copy到train和test文件夹
folder=r'E:\Shilong\murmur\03_circor_states'
copy_states_data(folder,absent_train_id,"\\Absent\\","\\train")
copy_states_data(folder,present_train_id,"\\Present\\","\\train")
copy_states_data(folder,absent_test_id,"\\Absent\\","\\test")
copy_states_data(folder,present_test_id,"\\Present\\","\\test")

# 保存train、test id为CSV文件
pd.DataFrame(absent_train_id).to_csv(absent_train_id_path, index=False, header=False)
pd.DataFrame(present_train_id).to_csv(present_train_id_path, index=False, header=False)
pd.DataFrame(absent_test_id).to_csv(absent_test_id_path, index=False, header=False)
pd.DataFrame(present_test_id).to_csv(present_test_id_path, index=False, header=False)