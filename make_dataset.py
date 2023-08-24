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
import soundfile as sf
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
def csv_reader_cl(file_name, clo_num):
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


# read the csv row_num-th row
def csv_reader_row(file_name, row_num):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        row = list(reader)
    return row[row_num]


# copy wav file to folder_path
def copy_file(src_path, folder_path, patient_id_list, mur, position):
    """将所有文件复制到目标目录"""
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)

            txtname = src_path + "\\" + patient_id + ".txt"
            wavname = src_path + "\\" + patient_id + pos + ".wav"
            heaname = src_path + "\\" + patient_id + pos + ".hea"
            tsvname = src_path + "\\" + patient_id + pos + ".tsv"

            if os.path.exists(txtname):
                shutil.copy(txtname, target_dir + "\\")
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir + "\\")
            if os.path.exists(heaname):
                shutil.copy(heaname, target_dir + "\\")
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir + "\\")


# copy wav file to folder_path
def copy_wav_file(src_path, folder_path, patient_id_list, mur, position):
    """将指定文件复制到目标目录"""
    count = 0
    # 1. make dir
    mur_dir = folder_path + "\\" + mur
    if not os.path.exists(mur_dir):
        os.makedirs(mur_dir)
    # 2. copy file
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)

            wavname = src_path + "\\" + patient_id + pos + ".wav"
            tsvname = src_path + "\\" + patient_id + pos + ".tsv"

            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir + "\\")
                count += 1
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir + "\\")
    print("copy file num: ", count)


# devide sounds into 4s segments
def pos_dir_make(dir_path, patient_id, pos):
    for po in pos:
        subdir = dir_path + patient_id + "\\" + patient_id + po
        wavname = subdir + ".wav"
        if os.path.exists(wavname):
            print("exist")
            mkdir(subdir)  # make dir


def index_load(tsvname):
    """读取tsv文件内容,不需要close函数"""
    with open(tsvname, "r") as f:
        txt_data = f.read()
    head = ["start", "end", "period"]
    data = txt_data.split("\n")[:-1]
    # 遍历每一行
    for l in data:
        sgmt = l.split("\t")
        if sgmt[2] != "0":
            head = np.vstack([head, sgmt])
    return head[1:]


# preprocessed PCGs were segmented into four heart sound states
def period_div(
    path,
    murmur,
    patient_id_list,
    positoin,
    id_data,
    Murmur_locations,
    Systolic_murmur_timing,
    Diastolic_murmur_timing,
):
    for mur in murmur:
        for patient_id in patient_id_list:
            for pos in positoin:
                dir_path = path + "//" + mur + patient_id + "//" + patient_id + pos
                tsv_path = dir_path + ".tsv"
                wav_path = dir_path + ".wav"

                index = id_data.index(patient_id)
                wav_location = pos[1:]  # 听诊区域
                locations = Murmur_locations[index].split("+")  # 有杂音的区域
                # 此听诊区有杂音
                if wav_location in locations:
                    Systolic_state = Systolic_murmur_timing[index]
                    Diastolic_state = Diastolic_murmur_timing[index]
                    # 没有 Systolic murmur
                    if Systolic_state == "nan":
                        Systolic_murmur = "Absent"
                    else:
                        Systolic_murmur = "Present"
                    # 没有 Diastolic murmur
                    if Diastolic_state == "nan":
                        Diastolic_murmur = "Absent"
                    else:
                        Diastolic_murmur = "Present"

                # 此听诊区没有杂音
                else:
                    Systolic_murmur = "Absent"
                    Diastolic_murmur = "Absent"
                    Systolic_state = "nan"
                    Diastolic_state = "nan"

                if Systolic_murmur == "Absent" and Diastolic_murmur == "Absent":
                    wav_murmur = "Absent"
                else:
                    wav_murmur = "Present"

                if Systolic_state == "nan" and Diastolic_state == "nan":
                    wav_state = "nan"
                else:
                    wav_state = Systolic_state+"+"+Diastolic_state

                if os.path.exists(tsv_path):
                    state_div_segment(  # 按照心音周期切割
                        tsv_path,
                        wav_path,
                        dir_path + "\\",
                        patient_id + pos,
                        wav_murmur,
                        wav_state
                    )

# preprocessed PCGs were segmented into four heart sound states


def state_div_segment(
    tsvname,
    wavname,
    state_path,
    index,
    wav_murmur,
    wav_state,

):
    index_file = index_load(tsvname)
    recording, fs = librosa.load(wavname, sr=4000)
    num = 0
    start_index = 0
    end_index = 0

    for i in range(index_file.shape[0] - 3):
        if index_file[i][2] == "1" and index_file[i + 3][2] == "4":
            start_index = float(index_file[i][0]) * fs
            end_index = float(index_file[i + 3][1]) * fs
            num = num + 1
            #  解决出现_0.wav的问题
            print(start_index, end_index)
            print("=============================================")
            print("wav name: " + wavname)
            buff = recording[int(start_index): int(end_index)]  # 字符串索引切割
            print("buff len: " + str(len(buff)))
            soundfile.write(
                state_path
                + "{}_{}_{}_{}.wav".format(
                    index, wav_murmur, wav_state, num
                ),
                buff,
                fs,
            )


# preprocessed PCGs were segmented into four heart sound states


def state_div(
    tsvname,
    wavname,
    state_path,
    index,
    Systolic_murmur,
    Diastolic_murmur,
    Systolic_state,
    Diastolic_state,
):
    index_file = index_load(tsvname)
    recording, fs = librosa.load(wavname, sr=4000)
    num = 0
    start_index2 = 0
    end_index2 = 0
    start_index4 = 0
    end_index4 = 0

    for i in range(index_file.shape[0] - 3):
        if index_file[i][2] == "2" and index_file[i + 2][2] == "4":
            start_index2 = float(index_file[i][0]) * fs
            end_index2 = float(index_file[i][1]) * fs
            start_index4 = float(index_file[i + 2][0]) * fs
            end_index4 = float(index_file[i + 2][1]) * fs
            num = num + 1
            #  解决出现_0.wav的问题
            print(start_index2, end_index2, start_index4, end_index4)
            print("=============================================")
            print("wav name: " + wavname)
            buff2 = recording[int(start_index2): int(end_index2)]  # 字符串索引切割
            buff4 = recording[int(start_index4): int(end_index4)]  # 字符串索引切割
            print("buff2 len: " + str(len(buff2)),
                  "buff4 len: " + str(len(buff4)))
            soundfile.write(
                state_path
                + "{}_{}_{}_{}_{}.wav".format(
                    index, "Systolic", num, Systolic_murmur, Systolic_state
                ),
                buff2,
                fs,
            )
            soundfile.write(
                state_path
                + "{}_{}_{}_{}_{}.wav".format(
                    index, "Diastolic", num, Diastolic_murmur, Diastolic_state
                ),
                buff4,
                fs,
            )


# get patient id from csv file
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row["0"] for row in reader]  # weight 同列的数据
        return id


# copy absent data to folder
def copy_states_data(folder, patient_id, murmur, type):
    din_path = folder + type + murmur
    if not os.path.exists(din_path):
        os.makedirs(din_path)
    for id in patient_id:
        dir_path = folder + murmur + id
        print(dir_path)
        for root, dir, file in os.walk(dir_path):
            for subdir in dir:
                subdir_path = os.path.join(root, subdir)
                print(subdir_path)
                # if os.path.exists(dir_path):
                shutil.copytree(subdir_path, din_path + subdir)

# 读取数据并打标签


def get_wav_data(dir_path):
    wav = []
    label = []
    # if not os.path.exists(csv_path):
    #     os.makedirs(csv_path)

    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 数据读取
                print("reading: " + subfile)
                y, sr = librosa.load(wav_path)
                y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                # print("y_16k size: "+y_16k.size)
                if y_16k.shape[0] < 15000:
                    y_16k = np.pad(
                        y_16k,
                        (0, 15000 - y_16k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                elif y_16k.shape[0] > 15000:
                    y_16k = y_16k[0:15000]
                wav.append(y_16k)
                # feature = pd.DataFrame(y_16k)
                # save_path = csv_path +"\\"+ subfile+ ".csv"
                # print("shape: "+feature.shape())
                # feature.to_csv(save_path, index=False, header=False)

                file_name = subfile.split("_")
                # 标签读取
                if file_name[2] == "Absent":  # Absent
                    label.append(0)
                if file_name[2] == "Present":  # Present
                    label.append(1)  # 说明该听诊区无杂音
    return np.array(wav), np.array(label)


def copy_wav2dataset(dir_path, setpath):
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            murmur = subfile.split("_")[2]
            subfile_path = os.path.join(root, subfile)
            print(subfile_path)

            if not os.path.exists(setpath+"\\present"):
                os.makedirs(setpath+"\\present")
            if not os.path.exists(setpath+"\\absent"):
                os.makedirs(setpath+"\\absent")

            if murmur == "Present":
                shutil.copy(subfile_path, setpath+"\\present")
            else:
                shutil.copy(subfile_path, setpath+"\\absent")

# 时域数据增强


def wav_time_stretch(path, time_path, speed_factor):
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    for root, dir, file in os.walk(path):
        for filename in file:
            print("processing "+filename)
            wav_path = os.path.join(root, filename)
            data, sr = librosa.load(wav_path, sr=4000)
            data_time_stretch = librosa.effects.time_stretch(
                data, rate=speed_factor)
            sf.write(os.path.join(time_path, filename +
                     f'time_{speed_factor}.wav'), data_time_stretch, sr)


def cal_len(dir_path, csv_path, name):
    alen = []
    plen = []
    # label=[]
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 数据读取
                print("reading: " + subfile)
                waveform, sr = librosa.load(wav_path)
                waveform_16k = librosa.resample(
                    y=waveform, orig_sr=sr, target_sr=16000)
                print("waveform_16k size: " + str(waveform_16k.size))

                if subfile.split("_")[2] == "Present":
                    plen.append(waveform_16k.size)
                else:
                    alen.append(waveform_16k.size)

    pd.DataFrame(plen).to_csv(
        csv_path+name+'//present.csv', index=False, header=False)
    pd.DataFrame(alen).to_csv(
        csv_path+name+'//absent.csv', index=False, header=False)

# ==================================================================== #
# ========================/ code executive /========================== #
# ==================================================================== #


csv_path = r"D:\Shilong\murmur\dataset_all\training_data.csv"
# get dataset tag from table
row_line = csv_reader_row(csv_path, 0)
tag_list = list()

# get index for 'Patient ID' and 'Outcome'
tag_list.append(row_line.index("Patient ID"))
tag_list.append(row_line.index("Murmur"))
tag_list.append(row_line.index("Murmur locations"))
tag_list.append(row_line.index("Systolic murmur timing"))
tag_list.append(row_line.index("Diastolic murmur timing"))

# for tag_index in tag_list:
id_data = csv_reader_cl(csv_path, tag_list[0])
Murmur = csv_reader_cl(csv_path, tag_list[1])
Murmur_locations = csv_reader_cl(csv_path, tag_list[2])
Systolic_murmur_timing = csv_reader_cl(csv_path, tag_list[3])
Diastolic_murmur_timing = csv_reader_cl(csv_path, tag_list[4])

# save data to csv file
Murmur_locations_path = r"D:\Shilong\murmur\03_circor_states\Murmur_locations.csv"
Systolic_murmur_timing_path = (
    r"D:\Shilong\murmur\03_circor_states\Systolic_murmur_timing.csv"
)
Diastolic_murmur_timing_path = (
    r"D:\Shilong\murmur\03_circor_states\Diastolic_murmur_timing.csv"
)
# save as csv file
# pd.DataFrame(Murmur_locations).to_csv(Murmur_locations_path, index=False, header=False)
# pd.DataFrame(Systolic_murmur_timing).to_csv(
#     Systolic_murmur_timing_path, index=False, header=False
# )
# pd.DataFrame(Diastolic_murmur_timing).to_csv(
#     Diastolic_murmur_timing_path, index=False, header=False
# )

# init aptient id list for absent present and unknown
absent_patient_id = list()
present_patient_id = list()

# get 'Absent' and 'Present' and 'Unknown' index
absent_id = [out for out, Murmur in enumerate(Murmur) if Murmur == "Absent"]
present_id = [out for out, Murmur in enumerate(Murmur) if Murmur == "Present"]

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
positoin = ["_AV", "_MV", "_PV", "_TV"]
murmur = ["Absent\\", "Present\\"]
period = ["s1", "systolic", "s2", "diastolic"]
folder_path = r"D:\Shilong\murmur\03_circor_states\period_try"

src_path = r"D:\Shilong\murmur\dataset_all\training_data"
# make dir and copy files for Present/Absent patients
# copy_wav_file(src_path, folder_path, absent_patient_id, "Absent", positoin)
# copy_wav_file(src_path, folder_path, present_patient_id, "Present", positoin)

# make dir for each position
# D:\Shilong\murmur\LM_wav_dataset

# for mur in murmur:
#     dir_path = folder_path + "//" + mur
#     for patient_id in absent_patient_id:
#         pos_dir_make(dir_path, patient_id, positoin)
#     for patient_id in present_patient_id:
#         pos_dir_make(dir_path, patient_id, positoin)

# 切数据，命名格式为：id+pos+state+num
# period_div(
#     folder_path,
#     murmur,
#     absent_patient_id,
#     positoin,
#     id_data,
#     Murmur_locations,
#     Systolic_murmur_timing,
#     Diastolic_murmur_timing,
# )
# period_div(
#     folder_path,
#     murmur,
#     present_patient_id,
#     positoin,
#     id_data,
#     Murmur_locations,
#     Systolic_murmur_timing,
#     Diastolic_murmur_timing,
# )

absent_train_id_path = r"D:\Shilong\murmur\03_circor_states\absent_train_id.csv"
absent_test_id_path = r"D:\Shilong\murmur\03_circor_states\absent_test_id.csv"
present_train_id_path = r"D:\Shilong\murmur\03_circor_states\present_train_id.csv"
present_test_id_path = r"D:\Shilong\murmur\03_circor_states\present_test_id.csv"

# 将absent_id和present_id按照8:2随机选取id划分为训练集和测试集
# absent_train_id=random.sample(absent_patient_id,int(len(absent_patient_id)*0.8))
# present_train_id=random.sample(present_patient_id,int(len(present_patient_id)*0.8))
# absent_test_id=list(set(absent_patient_id)-set(absent_train_id))
# present_test_id=list(set(present_patient_id)-set(present_train_id))
absent_train_id = csv_reader_cl(absent_train_id_path, 0)
absent_test_id = csv_reader_cl(absent_test_id_path, 0)
present_train_id = csv_reader_cl(present_train_id_path, 0)
present_test_id = csv_reader_cl(present_test_id_path, 0)
# 将训练集和测试集文件分别copy到train和test文件夹
folder = r"D:\Shilong\murmur\03_circor_states\period_try"
# copy_states_data(folder, absent_train_id, "\\Absent\\", "\\train")
# copy_states_data(folder, present_train_id, "\\Present\\", "\\train")
# copy_states_data(folder, absent_test_id, "\\Absent\\", "\\test")
# copy_states_data(folder, present_test_id, "\\Present\\", "\\test")

# 保存train、test id为CSV文件
# pd.DataFrame(absent_train_id).to_csv(absent_train_id_path, index=False, header=False)
# pd.DataFrame(present_train_id).to_csv(present_train_id_path, index=False, header=False)
# pd.DataFrame(absent_test_id).to_csv(absent_test_id_path, index=False, header=False)
# pd.DataFrame(present_test_id).to_csv(present_test_id_path, index=False, header=False)


# ========================/ parameteres define /========================== #
murmur_positoin = ["_AV", "_MV", "_PV", "_TV"]
murmur_ap = ["Absent\\", "Present\\"]
period = ["Systolic", "Diastolic"]

# ========================/ file path /========================== #
# get absent / present patient_id
id_data_path = r"D:\Shilong\murmur\03_circor_states\id_data.csv"
absent_csv_path = r"D:\Shilong\murmur\03_circor_states\absent_id.csv"
present_csv_path = r"D:\Shilong\murmur\03_circor_states\present_id.csv"
Diastolic_murmur_timing_path = (
    r"D:\Shilong\murmur\03_circor_states\Diastolic_murmur_timing.csv"
)
Systolic_murmur_timing_path = (
    r"D:\Shilong\murmur\03_circor_states\Systolic_murmur_timing.csv"
)
Murmur_locations_path = r"D:\Shilong\murmur\03_circor_states\Murmur_locations.csv"


filepath = r"D:\Shilong\murmur\03_circor_states"
absent_train_path = r'D:\Shilong\murmur\03_circor_states\period_try\train\Absent'
absent_test_path = r'D:\Shilong\murmur\03_circor_states\period_try\test\Absent'
present_train_path = r'D:\Shilong\murmur\03_circor_states\period_try\train\Present'
present_test_path = r'D:\Shilong\murmur\03_circor_states\period_try\test\Present'

# copy_wav2dataset(absent_train_path,
#                  r'D:\Shilong\murmur\03_circor_states\period_try\trainset')
# copy_wav2dataset(present_train_path,
#                  r'D:\Shilong\murmur\03_circor_states\period_try\trainset')
# copy_wav2dataset(absent_test_path,
#                  r'D:\Shilong\murmur\03_circor_states\period_try\testset')
# copy_wav2dataset(present_test_path,
#                  r'D:\Shilong\murmur\03_circor_states\period_try\testset')


folder = r"D:\Shilong\murmur\03_circor_statest"
npy_path = r"D:\Shilong\murmur\03_circor_states\npyFile"

path = r"D:\Shilong\murmur\03_circor_states\csv"
train_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
test_path = r"D:\Shilong\murmur\03_circor_states\test_csv"


absent_train_csv_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
absent_test_csv_path = r"D:\Shilong\murmur\03_circor_states\test_csv"
present_train_csv_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
present_test_csv_path = r"D:\Shilong\murmur\03_circor_states\test_csv"

filepath = r"D:\Shilong\murmur\03_circor_states"
absent_train_path = r"D:\Shilong\murmur\03_circor_states\period_try\trainset\absent"
absent_test_path = r"D:\Shilong\murmur\03_circor_states\period_try\testset\absent"
present_train_path = r"D:\Shilong\murmur\03_circor_states\period_try\trainset\present"
present_test_path = r"D:\Shilong\murmur\03_circor_states\period_try\testset\present"
present_train_path_8 = r"D:\Shilong\murmur\03_circor_states\period_try\trainset\time_stretch0.8"
present_train_path_12 = r"D:\Shilong\murmur\03_circor_states\period_try\trainset\time_stretch1.2"
npy_path_padded = r"D:\Shilong\murmur\03_circor_states\period_try\npyFile"

# wav_time_stretch(present_train_path, present_train_path_8, 0.8)
# wav_time_stretch(present_train_path, present_train_path_12, 1.2)
# ========================/ get lists /========================== #
# id_data = get_patientid(id_data_path)
# absent_patient_id = get_patientid(absent_csv_path)
# present_patient_id = get_patientid(present_csv_path)
# Diastolic_murmur_timing = get_patientid(Diastolic_murmur_timing_path)
# Systolic_murmur_timing = get_patientid(Systolic_murmur_timing_path)
# Murmur_locations = get_patientid(Murmur_locations_path)

csv_path = r"D:\Shilong\murmur\03_circor_states\period_try\csv"
cal_len(absent_train_path, csv_path, 'at')
cal_len(absent_test_path, csv_path, 'av')
cal_len(present_train_path, csv_path, 'pt')
cal_len(present_test_path, csv_path, 'pv')
cal_len(present_train_path_8, csv_path, 'pt8')
cal_len(present_train_path_12, csv_path, 'pt12')

absent_train_features, absent_train_label = get_wav_data(
    absent_train_path
)  # absent
absent_test_features, absent_test_label = get_wav_data(
    absent_test_path
)  # absent
present_train_features, present_train_label = get_wav_data(
    present_train_path
)  # present
present_test_features, present_test_label = get_wav_data(
    present_test_path
)  # present
present_train_features_8, present_train_label_8 = get_wav_data(
    present_train_path_8
)  # present
present_train_features_12, present_train_label_12 = get_wav_data(
    present_train_path_12
)  # present


# # # ========================/ save as npy file /========================== #
np.save(npy_path_padded + r"\absent_train_features.npy", absent_train_features)
np.save(npy_path_padded + r"\absent_test_features.npy", absent_test_features)
np.save(npy_path_padded + r"\present_train_features.npy", present_train_features)
np.save(npy_path_padded + r"\present_test_features.npy", present_test_features)

np.save(npy_path_padded + r"\absent_train_label.npy", absent_train_label)
np.save(npy_path_padded + r"\absent_test_label.npy", absent_test_label)
np.save(npy_path_padded + r"\present_train_label.npy", present_train_label)
np.save(npy_path_padded + r"\present_test_label.npy", present_test_label)

np.save(npy_path_padded + r"\present_train_features_8.npy",
        present_train_features_8)
np.save(npy_path_padded + r"\present_train_features_12.npy",
        present_train_features_12)
np.save(npy_path_padded + r"\present_train_label_8.npy", present_train_label_8)
np.save(npy_path_padded + r"\present_train_label_12.npy", present_train_label_12)
