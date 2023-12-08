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
from BEATs_def import mkdir, csv_reader_cl, csv_reader_row
import soundfile as sf
from BEATs_def import get_wav_data, get_patientid
from dataAugmentation import data_Auge
import pandas as pd
from helper_code import *
# ========================/ functions define /========================== #


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


def copy_wav_file(src_path, folder_path, patient_id_list, mur, position):
    """将wav和tsv文件复制到目标目录"""
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
            txtname = src_path + "\\" + patient_id + ".txt"
            if os.path.exists(wavname):
                shutil.copy(wavname, target_dir + "\\")
                count += 1
            if os.path.exists(tsvname):
                shutil.copy(tsvname, target_dir + "\\")
            if os.path.exists(txtname):
                shutil.copy(txtname, target_dir + "\\")
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
    mur,
    patient_id_list,
    positoin,
    id_data,
    Murmur_locations,
    Systolic_murmur_timing,
    Diastolic_murmur_timing,
):

    for patient_id in patient_id_list:
        patient_dir_path = path + mur + patient_id + "\\" + patient_id
        txtpath = patient_dir_path+".txt"
        current_patient_data = load_patient_data(txtpath)
        hunman_feat = get_features_mod(current_patient_data)
        for pos in positoin:
            dir_path = path + mur + patient_id + "\\" + patient_id + pos
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
            # 如果是present的有杂音区域，或absent区域
            # if (mur == "Absent\\") or (mur == "Present\\" and (wav_location in locations)):
            if os.path.exists(tsv_path):
                state_div(
                    tsv_path,
                    wav_path,
                    dir_path + "\\",
                    patient_id + pos,
                    Systolic_murmur,
                    Diastolic_murmur,
                    Systolic_state,
                    Diastolic_state,
                    hunman_feat
                )


def state_div(
    tsvname,
    wavname,
    state_path,
    index,
    Systolic_murmur,
    Diastolic_murmur,
    Systolic_state,
    Diastolic_state,
    hunman_feat
):
    """切割出s1+收缩和s2+舒张"""
    index_file = index_load(tsvname)
    recording, fs = librosa.load(wavname, sr=4000)
    num = 0
    # start_index1 = 0
    # end_index1 = 0
    # start_index2 = 0
    # end_index2 = 0
    # count = 0
    for i in range(index_file.shape[0] - 3):
        # if count == 20:
        #     break
        if index_file[i][2] == "1" and index_file[i + 2][2] == "3":
            start_index1 = float(index_file[i][0]) * fs
            end_index1 = float(index_file[i+1][1]) * fs
            start_index2 = float(index_file[i + 2][0]) * fs
            end_index2 = float(index_file[i + 3][1]) * fs
            num = num + 1

            #  解决出现_0.wav的问题
            print(start_index1, end_index1, start_index2, end_index2)
            print("=============================================")
            print("wav name: " + wavname)
            buff1 = recording[int(start_index1): int(end_index1)]  # 字符串索引切割
            buff2 = recording[int(start_index2): int(end_index2)]  # 字符串索引切割
            print("buff1 len: " + str(len(buff1)),
                  "buff2 len: " + str(len(buff2)))
            # if Systolic_murmur == "Present" and Diastolic_murmur == "Absent":
            #     # 切收缩期
            #     soundfile.write(
            #         state_path
            #         + f"{index}_s1+Systolic_{num}_{Systolic_murmur}_{Systolic_state}_{hunman_feat}.wav",
            #         buff1,
            #         fs,
            #     )
            # else:
            # 切收缩期
            soundfile.write(
                state_path
                + f"{index}_s1+Systolic_{num}_{Systolic_murmur}_{Systolic_state}_{hunman_feat}.wav",
                buff1,
                fs,
            )
            # 切舒张期
            soundfile.write(
                state_path
                + f"{index}_s2+Diastolic_{num}_{Diastolic_murmur}_{Diastolic_state}_{hunman_feat}.wav",
                buff2,
                fs,
            )
            # count += 1


def state_div2(
    tsvname,
    wavname,
    state_path,
    index,
    Systolic_murmur,
    Diastolic_murmur,
    Systolic_state,
    Diastolic_state,
):
    """切割出收缩期和舒张期"""
    index_file = index_load(tsvname)
    recording, fs = librosa.load(wavname, sr=4000)
    num1 = 0
    num2 = 0
    # start_index1 = 0
    # end_index1 = 0
    # start_index2 = 0
    # end_index2 = 0

    for i in range(index_file.shape[0]):
        if index_file[i][2] == "2":
            start_index1 = float(index_file[i][0]) * fs
            end_index1 = float(index_file[i][1]) * fs
            num1 = num1 + 1
            buff1 = recording[int(start_index1): int(end_index1)]  # 字符串索引切割
            print(start_index1, end_index1)
            print("buff1 len: " + str(len(buff1)))
            soundfile.write(
                state_path
                + "{}_{}_{}_{}_{}.wav".format(
                    index, "Systolic", num1, Systolic_murmur, Systolic_state
                ),
                buff1,
                fs,
            )

        if index_file[i][2] == "4":
            start_index2 = float(index_file[i][0]) * fs
            end_index2 = float(index_file[i][1]) * fs
            num2 = num2 + 1
            buff2 = recording[int(start_index2): int(end_index2)]  # 字符串索引切割
            print(start_index2, end_index2)
            print("buff2 len: " + str(len(buff2)))
            soundfile.write(
                state_path
                + "{}_{}_{}_{}_{}.wav".format(
                    index, "Diastolic", num2, Diastolic_murmur, Diastolic_state
                ),
                buff2,
                fs,
            )


# get patient id from csv file
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row["0"] for row in reader]  # weight 同列的数据
        return id


# copy data to folder
def copy_states_data(patient_id, folder, type, murmur):
    traget_path = folder+type+murmur
    if not os.path.exists(traget_path):
        os.makedirs(traget_path)
    for id in patient_id:
        dir_path = folder + murmur + id
        print(dir_path)
        for root, dir, file in os.walk(dir_path):
            for subdir in dir:
                subdir_path = os.path.join(root, subdir)
                print(subdir_path)
                if os.path.exists(dir_path):
                    shutil.copytree(subdir_path, traget_path + subdir)
                else:
                    print("dir not exist")


def fold_devide(data, flod_num=5):
    """五折交叉验证
    将输入列表打乱，然后分成五份
    output: flod5 = {0:[],1:[],2:[],3:[],4:[]}
    """
    # 打乱序列
    random.shuffle(data)
    # 五折
    flod5 = {}
    point = []
    for i in range(flod_num):
        point.append(i*round(len(data)/flod_num))
    # print(point)
    # 分割序列
    for i in range(len(point)):
        if i < len(point)-1:
            flod5[i] = []
            flod5[i].extend(data[point[i]:point[i+1]])
        else:
            flod5[i] = []
            flod5[i].extend(data[point[-1]:])
    return flod5


def data_set(root_path):
    """数据增强，包括时间拉伸和反转"""
    # root_path = r"D:\Shilong\murmur\01_dataset\06_new5fold"
    npy_path_padded = root_path+r"\npyFile_padded\npy_files01"
    index_path = root_path + r"\npyFile_padded\index_files01"
    if not os.path.exists(npy_path_padded):
        os.makedirs(npy_path_padded)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    for k in range(5):
        src_fold_root_path = root_path+r"\fold_set_"+str(k)
        # data_Auge(src_fold_root_path)
        for folder in os.listdir(src_fold_root_path):
            dataset_path = os.path.join(src_fold_root_path, folder)
            if k == 0 and folder == "absent":
                features, label, names, index, data_id, feat = get_wav_data(
                    dataset_path, num=0)  # absent
            else:
                features, label, names, index, data_id, feat = get_wav_data(
                    dataset_path, data_id)  # absent
            np.save(npy_path_padded +
                    f"\\{folder}_features_norm01_fold{k}.npy", features)
            np.save(npy_path_padded +
                    f"\\{folder}_labels_norm01_fold{k}.npy", label)
            np.save(npy_path_padded +
                    f"\\{folder}_index_norm01_fold{k}.npy", index)
            np.save(npy_path_padded +
                    f"\\{folder}_name_norm01_fold{k}.npy", names)
            np.save(npy_path_padded +
                    f"\\{folder}_feat_norm01_fold{k}.npy", feat)
            absent_train_dic = zip(index, names, feat)
            pd.DataFrame(absent_train_dic).to_csv(
                index_path+f"\\fold{k}_{folder}_disc.csv", index=False, header=False)
    print("data set done!")


def get_features_mod(data):
    # Extract the age group, sex and the pregnancy status features
    age_group = get_age(data)
    age_list = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    is_pregnant = get_pregnancy_status(data)
    if age_group not in ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']:
        if is_pregnant:
            age = 'Young Adult'
        else:
            age = 'Child'
    else:
        age = age_group
    age_fea = str(age_list.index(age))
    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)
    if compare_strings(sex, 'Female'):
        sex_features = '0'
    elif compare_strings(sex, 'Male'):
        sex_features = '1'
    if is_pregnant:
        preg_fea = '1'
    else:
        preg_fea = '0'
    return age_fea + sex_features + preg_fea


# ==================================================================== #
# ========================/ code executive /========================== #
# ==================================================================== #
if __name__ == '__main__':
    csv_path = r"D:\Shilong\murmur\dataset_all\training_data.csv"
    # get dataset tag from table
    row_line = csv_reader_row(csv_path, 0)
    tag_list = []

    # get index for 'Patient ID' and 'Outcome'
    tag_list.append(row_line.index("Patient ID"))
    tag_list.append(row_line.index("Murmur"))
    tag_list.append(row_line.index("Murmur locations"))
    tag_list.append(row_line.index("Systolic murmur timing"))
    tag_list.append(row_line.index("Diastolic murmur timing"))

    # # for tag_index in tag_list:
    id_data = csv_reader_cl(csv_path, tag_list[0])
    Murmur = csv_reader_cl(csv_path, tag_list[1])
    Murmur_locations = csv_reader_cl(csv_path, tag_list[2])
    Systolic_murmur_timing = csv_reader_cl(csv_path, tag_list[3])
    Diastolic_murmur_timing = csv_reader_cl(csv_path, tag_list[4])
    # TODO 修改此处的root_path
    root_path = r"D:\Shilong\murmur\01_dataset\11_baseset"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    # save data to csv file
    Murmur_locations_path = root_path+r"\Murmur_locations.csv"
    Systolic_murmur_timing_path = (
        root_path+r"\Systolic_murmur_timing.csv"
    )
    Diastolic_murmur_timing_path = (
        root_path+r"\Diastolic_murmur_timing.csv"
    )
    pd.DataFrame(Murmur_locations).to_csv(
        Murmur_locations_path, index=False, header=False)
    pd.DataFrame(Systolic_murmur_timing).to_csv(
        Systolic_murmur_timing_path, index=False, header=False
    )
    pd.DataFrame(Diastolic_murmur_timing).to_csv(
        Diastolic_murmur_timing_path, index=False, header=False
    )

    # init aptient id list for absent present and unknown
    absent_patient_id = []
    present_patient_id = []
    # get 'Absent' and 'Present' and 'Unknown' index
    absent_id = [out for out, Murmur in enumerate(
        Murmur) if Murmur == "Absent"]
    present_id = [out for out, Murmur in enumerate(
        Murmur) if Murmur == "Present"]

    # get 'Absent' and 'Present' and 'Unknown' patients ID
    for id in absent_id:
        absent_patient_id.append(id_data[id])
    for id in present_id:
        present_patient_id.append(id_data[id])

    absent_id_path = root_path+r"\absent_id.csv"
    patient_id_path = root_path+r"\patient_id.csv"
    # 对Present和Absent分五折
    # save patient id as csv
    pd.DataFrame(data=absent_patient_id, index=None).to_csv(
        absent_id_path, index=False, header=False)
    pd.DataFrame(data=present_patient_id, index=None).to_csv(
        patient_id_path, index=False, header=False)
    fold_absent = fold_devide(absent_patient_id)
    fold_present = fold_devide(present_patient_id)
    # 对Present和Absent分五折
    # 分别保存每折的id
    for k, v in fold_absent.items():
        pd.DataFrame(data=v, index=None).to_csv(
            root_path+r"\absent_fold_"+str(k)+".csv", index=False, header=False)
    for k, v in fold_present.items():
        pd.DataFrame(data=v, index=None).to_csv(
            root_path+r"\present_fold_"+str(k)+".csv", index=False, header=False)
    # digaiation position
    # define path options
    positoin = ["_AV", "_MV", "_PV", "_TV"]
    murmur = ["Absent\\", "Present\\"]
    period = ["s1", "systolic", "s2", "diastolic"]
    src_path = r"D:\Shilong\murmur\dataset_all\training_data"
    folder_path = root_path+"\\"
    # 将wav文件和tsv文件copy到目标文件夹
    copy_wav_file(src_path, folder_path, absent_patient_id, "Absent", positoin)
    copy_wav_file(src_path, folder_path,
                  present_patient_id, "Present", positoin)

    src_path = r"D:\Shilong\murmur\dataset_all\training_data"
    # 创建每个wav文件的文件夹
    for mur in murmur:
        dir_path = folder_path + mur
        for patient_id in absent_patient_id:
            pos_dir_make(dir_path, patient_id, positoin)
        for patient_id in present_patient_id:
            pos_dir_make(dir_path, patient_id, positoin)

    # 切数据，命名格式为：id+pos+state+num
    # 对数据打标签
    # absent
    period_div(
        folder_path,
        "Absent\\",
        absent_patient_id,
        positoin,
        id_data,
        Murmur_locations,
        Systolic_murmur_timing,
        Diastolic_murmur_timing,
    )
    # present
    period_div(
        folder_path,
        "Present\\",
        present_patient_id,
        positoin,
        id_data,
        Murmur_locations,
        Systolic_murmur_timing,
        Diastolic_murmur_timing,
    )

    absent_train_id_path = root_path+r"\absent_train_id.csv"
    absent_test_id_path = root_path+r"\absent_test_id.csv"
    present_train_id_path = root_path+r"\present_train_id.csv"
    present_test_id_path = root_path+r"\present_test_id.csv"

    # 将absent_id和present_id按照8:2随机选取id划分为训练集和测试集
    # absent_train_id = random.sample(
    #     absent_patient_id, int(len(absent_patient_id)*0.8))
    # present_train_id = random.sample(
    #     present_patient_id, int(len(present_patient_id)*0.8))
    # absent_test_id = list(set(absent_patient_id)-set(absent_train_id))
    # present_test_id = list(set(present_patient_id)-set(present_train_id))

    # 将训练集和测试集文件分别copy到train和test文件夹
    # copy_states_data(absent_train_id, root_path, "\\train", "\\Absent\\")
    # copy_states_data(present_train_id,root_path,  "\\train", "\\Present\\")
    # copy_states_data(absent_test_id,root_path,  "\\test", "\\Absent\\")
    # copy_states_data( present_test_id,root_path, "\\test", "\\Present\\")

    # 按照每折的id复制数据到每折对应文件夹
    # 此处执行后的数据，数据只按折分开了，并没有按present和Absnet分开
    for k, v in fold_absent.items():
        copy_states_data(v, root_path, "\\fold_"+str(k), "\\Absent\\")
    for k, v in fold_present.items():
        copy_states_data(v, root_path, "\\fold_"+str(k), "\\Present\\")
    # 若要继续按照present和absent分开，需要在save_as_npy.py中修改代码

    # 保存train、test id为CSV文件
    # pd.DataFrame(absent_train_id).to_csv(
    #     absent_train_id_path, index=False, header=False)
    # pd.DataFrame(present_train_id).to_csv(
    #     present_train_id_path, index=False, header=False)
    # pd.DataFrame(absent_test_id).to_csv(
    #     absent_test_id_path, index=False, header=False)
    # pd.DataFrame(present_test_id).to_csv(
    #     present_test_id_path, index=False, header=False)

    # # 读取训练集和测试集id划分
    # absent_train_id = csv_reader_cl(absent_train_id_path, 0)
    # absent_test_id = csv_reader_cl(absent_test_id_path, 0)
    # present_train_id = csv_reader_cl(present_train_id_path, 0)
    # present_test_id = csv_reader_cl(present_test_id_path, 0)

    # ========================/ get lists /========================== #
    # root_path = r"D:\Shilong\murmur\01_dataset\05_5fold"
    # file_path_train = r'D:\Shilong\murmur\01_dataset\05_5fold\train'
    # file_path_test = r'D:\Shilong\murmur\01_dataset\05_5fold\test'
    # fold_path:rootpath+\fold_0\Absent or Present
    # target_dir_train_a = root_path+r'\trainset\absent'
    # target_dir_train_p = root_path+r'\trainset\present'
    # target_dir_test_a = root_path+r'\testset\absent'
    # target_dir_test_p = root_path+r'\testset\present'

    for k in range(5):
        for murmur in ['Absent', 'Present']:
            src_fold_path = root_path+r"\fold_"+str(k)+"\\"+murmur+"\\"
            target_dir = root_path+r'\fold_set_'+str(k)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not os.path.exists(target_dir + "\\absent\\"):
                os.makedirs(target_dir + "\\absent\\")
            if not os.path.exists(target_dir + "\\present\\"):
                os.makedirs(target_dir + "\\present\\")
            for root, dir, file in os.walk(src_fold_path):
                for subfile in file:
                    files = os.path.join(root, subfile)
                    print(subfile)
                    state = subfile.split("_")[4]
                    if state == 'Absent':
                        shutil.copy(files, target_dir + "\\absent\\")
                    elif state == 'Present':
                        shutil.copy(files, target_dir + "\\present\\")
                    else:
                        raise ValueError("state error")

    for k in range(5):
        src_fold_root_path = root_path+r"'\fold_set_"+str(k)
        for murmur in ['absent', 'present']:
            src_fold_path = src_fold_root_path+"\\"+murmur+"\\"

data_set(root_path)


# # 复制到trainset和testset
# # trainset
# 将训练集和测试集文件分别copy到train和test文件夹
# for root, dir, file in os.walk(file_path_train):
#     for subfile in file:
#         files = os.path.join(root, subfile)
#         print(subfile)
#         state = subfile.split("_")[4]
#         if state == 'Absent':
#             shutil.copy(files, target_dir_train_a + "\\")
#         if state == 'Present':
#             shutil.copy(files, target_dir_train_p + "\\")
# # testset
# for root, dir, file in os.walk(file_path_test):
#     for subfile in file:
#         files = os.path.join(root, subfile)
#         print(subfile)
#         state = subfile.split("_")[4]
#         if state == 'Absent':
#             shutil.copy(files, target_dir_test_a + "\\")
#         if state == 'Present':
#             shutil.copy(files, target_dir_test_p + "\\")
