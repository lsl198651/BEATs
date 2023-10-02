import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from BEATs_def import get_wav_data, get_patientid
import pandas as pd


# ========================/ parameteres define /========================== #
murmur_positoin = ["_AV", "_MV", "_PV", "_TV"]
murmur_ap = ["Absent\\", "Present\\"]
period = ["Systolic", "Diastolic"]

# ========================/ get lists /========================== #
dataset_path = r'D:\Shilong\murmur\01_dataset\01_s1s2'
file_path_train = r'D:\Shilong\murmur\01_dataset\01_s1s2\train'
file_path_test = r'D:\Shilong\murmur\01_dataset\01_s1s2\test'
target_dir_train_a = dataset_path+r'\trainset\absent'
target_dir_train_p = dataset_path+r'\trainset\present'
target_dir_test_a = dataset_path+r'\testset\absent'
target_dir_test_p = dataset_path+r'\testset\present'

if not os.path.exists(target_dir_train_a):
    os.makedirs(target_dir_train_a)
if not os.path.exists(target_dir_train_p):
    os.makedirs(target_dir_train_p)
if not os.path.exists(target_dir_test_a):
    os.makedirs(target_dir_test_a)
if not os.path.exists(target_dir_test_p):
    os.makedirs(target_dir_test_p)
# 复制到trainset和testset
# for root, dir, file in os.walk(file_path_train):
#     for subfile in file:
#         files = os.path.join(root, subfile)
#         print(subfile)
#         state = subfile.split("_")[4]
#         if state == 'Absent':
#             shutil.copy(files, target_dir_train_a + "\\")
#         if state == 'Present':
#             shutil.copy(files, target_dir_train_p + "\\")

# for root, dir, file in os.walk(file_path_test):
#     for subfile in file:
#         files = os.path.join(root, subfile)
#         print(subfile)
#         state = subfile.split("_")[4]
#         if state == 'Absent':
#             shutil.copy(files, target_dir_test_a + "\\")
#         if state == 'Present':
#             shutil.copy(files, target_dir_test_p + "\\")


# ========================/ Data Augementation /========================== #
# 数据增强文件
speed_factor1 = 1.1
speed_factor0 = 0.8
time_path1 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch0.9'
time_path2 = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\time_stretch1.2'
path = r'D:\Shilong\murmur\01_dataset\01_s1s2\trainset\present'
if not os.path.exists(time_path1):
    os.makedirs(time_path1)
if not os.path.exists(time_path2):
    os.makedirs(time_path2)

# for root, dir, file in os.walk(path):
#     for filename in file:
#         print("processing "+filename)
#         wav_path = os.path.join(root, filename)
#         data, sr = librosa.load(wav_path, sr=4000)

#         data_time_stretch = librosa.effects.time_stretch(
#             data, rate=speed_factor1)
#         sf.write(os.path.join(time_path1, filename+'_.wav'), data, sr)

#         data_time_stretch = librosa.effects.time_stretch(
#             data, rate=speed_factor0)
#         sf.write(os.path.join(time_path2, filename+'_.wav'), data, sr)

# ========================/ file path /========================== #
# get absent / present patient_id
csv_folder = r"D:\Shilong\murmur\03_circor_statest"
id_data_path = csv_folder+r"\id_data.csv"
absent_csv_path = csv_folder+r"\absent_id.csv"
present_csv_path = csv_folder+r"\present_id.csv"
Diastolic_murmur_timing_path = (
    csv_folder+r"\Diastolic_murmur_timing.csv"
)
Systolic_murmur_timing_path = (
    csv_folder+r"\Systolic_murmur_timing.csv"
)
Murmur_locations_path = csv_folder+r"\Murmur_locations.csv"

wav_filepath = r"D:\Shilong\murmur\01_dataset\01_s1s2"
absent_train_path = wav_filepath+r"\trainset\absent"
absent_test_path = wav_filepath+r"\testset\absent"
present_train_path = wav_filepath+r"\trainset\present"
present_test_path = wav_filepath+r"\testset\present"
present_train_path_8 = wav_filepath+r"\trainset\time_stretch0.8"
present_train_path_9 = wav_filepath+r"\trainset\time_stretch0.9"
present_train_path_11 = wav_filepath+r"\trainset\time_stretch1.1"
present_train_path_12 = wav_filepath+r"\trainset\time_stretch1.2"
present_train_path_reverse = wav_filepath+r"\trainset\reverse"
present_train_path_reverse8 = wav_filepath+r"\trainset\reverse0.8"
present_train_path_reverse9 = wav_filepath+r"\trainset\reverse0.9"
present_train_path_reverse11 = wav_filepath+r"\trainset\reverse1.1"
present_train_path_reverse12 = wav_filepath+r"\trainset\reverse1.2"


# ========================/ get lists /========================== #


absent_train_features, absent_train_label, absent_train_names, absent_train_index, data_id = get_wav_data(
    absent_train_path
)  # absent
absent_test_features, absent_test_label, absent_test_names, absent_test_index, data_id = get_wav_data(
    absent_test_path, data_id
)  # absent
present_train_features, present_train_label, present_train_names, present_train_index, data_id = get_wav_data(
    present_train_path, data_id
)  # present
present_test_features, present_test_label, present_test_names, present_test_index, data_id = get_wav_data(
    present_test_path, data_id
)  # present

# 保存增强后的特征和标签
# x 0.8
present_train_features_8, present_train_label_8, present_train_names_8, present_train_index_8, data_id = get_wav_data(
    present_train_path_8, data_id
)
# x 0.9
present_train_features_9, present_train_label_9, present_train_names_9, present_train_index_9, data_id = get_wav_data(
    present_train_path_9, data_id
)
# x 1.1
present_train_features_11, present_train_label_11, present_train_names_11, present_train_index_11, data_id = get_wav_data(
    present_train_path_11, data_id)
# x 1.2
present_train_features_12, present_train_label_12, present_train_names_12, present_train_index_12, data_id = get_wav_data(
    present_train_path_12, data_id
)
# 反转后的特征和标签
present_train_features_reverse, present_train_label_reverse, present_train_names_reverse, present_train_index_reverse,  data_id = get_wav_data(
    present_train_path_reverse, data_id
)  # present
present_train_features_reverse8, present_train_label_reverse8, present_train_names_reverse8, present_train_index_reverse8, data_id = get_wav_data(
    present_train_path_reverse8, data_id
)  # present
present_train_features_reverse9, present_train_label_reverse9, present_train_names_reverse9, present_train_index_reverse9, data_id = get_wav_data(
    present_train_path_reverse9, data_id
)  # present
present_train_features_reverse11, present_train_label_reverse11, present_train_names_reverse11, present_train_index_reverse11, data_id = get_wav_data(
    present_train_path_reverse11, data_id
)  # present
present_train_features_reverse12, present_train_label_reverse12, present_train_names_reverse12, present_train_index_reverse12, data_id = get_wav_data(
    present_train_path_reverse12, data_id
)  # present


# ========================/ 保存数据 /========================== #
# -------------------------/ 保存特征数据 /------------------------- #
npy_path_padded = wav_filepath+r"\npyFile_padded\normalized\list_npy_files"
if not os.path.exists(npy_path_padded):
    os.makedirs(npy_path_padded)

np.save(npy_path_padded + r"\absent_train_features_norm.npy",
        absent_train_features)
np.save(npy_path_padded + r"\absent_test_features_norm.npy", absent_test_features)
np.save(npy_path_padded + r"\present_train_features_norm.npy",
        present_train_features)
np.save(npy_path_padded + r"\present_test_features_norm.npy",
        present_test_features)
# 数据增强部分
np.save(npy_path_padded + r"\present_train_features_8_norm.npy",
        present_train_features_8)
np.save(npy_path_padded + r"\present_train_features_9_norm.npy",
        present_train_features_9)
np.save(npy_path_padded + r"\present_train_features_11_norm.npy",
        present_train_features_11)
np.save(npy_path_padded + r"\present_train_features_12_norm.npy",
        present_train_features_12)
np.save(npy_path_padded + r"\present_train_features_reverse_norm.npy",
        present_train_features_reverse)
np.save(npy_path_padded + r"\present_train_features_reverse8_norm.npy",
        present_train_features_reverse8)
np.save(npy_path_padded + r"\present_train_features_reverse9_norm.npy",
        present_train_features_reverse9)
np.save(npy_path_padded + r"\present_train_features_reverse11_norm.npy",
        present_train_features_reverse11)
np.save(npy_path_padded + r"\present_train_features_reverse12_norm.npy",
        present_train_features_reverse12)

# -------------------------/ 保存标签数据 /------------------------- #
np.save(npy_path_padded + r"\absent_train_label_norm.npy", absent_train_label)
np.save(npy_path_padded + r"\absent_test_label_norm.npy", absent_test_label)
np.save(npy_path_padded + r"\present_train_label_norm.npy", present_train_label)
np.save(npy_path_padded + r"\present_test_label_norm.npy", present_test_label)

np.save(npy_path_padded + r"\present_train_label_8_norm.npy",
        present_train_label_8)
np.save(npy_path_padded + r"\present_train_label_9_norm.npy",
        present_train_label_9)
np.save(npy_path_padded + r"\present_train_label_11_norm.npy",
        present_train_label_11)
np.save(npy_path_padded + r"\present_train_label_12_norm.npy",
        present_train_label_11)

np.save(npy_path_padded + r"\present_train_label_reverse_norm.npy",
        present_train_label_reverse)
np.save(npy_path_padded + r"\present_train_label_reverse8_norm.npy",
        present_train_label_reverse8)
np.save(npy_path_padded + r"\present_train_label_reverse9_norm.npy",
        present_train_label_reverse9)
np.save(npy_path_padded + r"\present_train_label_reverse11_norm.npy",
        present_train_label_reverse11)
np.save(npy_path_padded + r"\present_train_label_reverse12_norm.npy",
        present_train_label_reverse12)

# -------------------------/ 保存索引数据 /------------------------- #
np.save(npy_path_padded + r"\absent_train_index_norm.npy", absent_train_index)
np.save(npy_path_padded + r"\absent_test_index_norm.npy", absent_test_index)
np.save(npy_path_padded + r"\present_train_index_norm.npy", present_train_index)
np.save(npy_path_padded + r"\present_test_index_norm.npy", present_test_index)
np.save(npy_path_padded + r"\present_train_index_8_norm.npy",
        present_train_index_8)
np.save(npy_path_padded + r"\present_train_index_9_norm.npy",
        present_train_index_9)
np.save(npy_path_padded + r"\present_train_index_11_norm.npy",
        present_train_index_11)
np.save(npy_path_padded + r"\present_train_index_12_norm.npy",
        present_train_index_11)
np.save(npy_path_padded + r"\present_train_index_reverse_norm.npy",
        present_train_index_reverse)
np.save(npy_path_padded + r"\present_train_index_reverse8_norm.npy",
        present_train_index_reverse8)
np.save(npy_path_padded + r"\present_train_index_reverse9_norm.npy",
        present_train_index_reverse9)
np.save(npy_path_padded + r"\present_train_index_reverse11_norm.npy",
        present_train_index_reverse11)
np.save(npy_path_padded + r"\present_train_index_reverse12_norm.npy",
        present_train_index_reverse12)


# -------------------------/ 保存名称数据 /------------------------- #
np.save(npy_path_padded + r"\absent_train_names_norm.npy", absent_train_names)
np.save(npy_path_padded + r"\absent_test_names_norm.npy", absent_test_names)
np.save(npy_path_padded + r"\present_train_names_norm.npy", present_train_names)
np.save(npy_path_padded + r"\present_test_names_norm.npy", present_test_names)
np.save(npy_path_padded + r"\present_train_names_8_norm.npy",
        present_train_names_8)
np.save(npy_path_padded + r"\present_train_names_9_norm.npy",
        present_train_names_9)
np.save(npy_path_padded + r"\present_train_names_11_norm.npy",
        present_train_names_11)
np.save(npy_path_padded + r"\present_train_names_12_norm.npy",
        present_train_names_11)
np.save(npy_path_padded + r"\present_train_names_reverse_norm.npy",
        present_train_names_reverse)
np.save(npy_path_padded + r"\present_train_names_reverse8_norm.npy",
        present_train_names_reverse8)
np.save(npy_path_padded + r"\present_train_names_reverse9_norm.npy",
        present_train_names_reverse9)
np.save(npy_path_padded + r"\present_train_names_reverse11_norm.npy",
        present_train_names_reverse11)
np.save(npy_path_padded + r"\present_train_names_reverse12_norm.npy",
        present_train_names_reverse12)

index_path = r"D:\Shilong\murmur\01_dataset\01_s1s2\npyFile_padded\normalized\index_files"
if not os.path.exists(index_path):
    os.makedirs(index_path)
# 生成字典
absent_train_dic = zip(absent_train_index, absent_train_names)
absent_test_dic = zip(absent_test_index, absent_test_names)
present_train_dic = zip(present_train_index, present_train_names)
present_test_dic = zip(present_test_index, present_test_names)
present_train_dic_8 = zip(present_train_index_8, present_train_names_8)
present_train_dic_9 = zip(present_train_index_9, present_train_names_9)
present_train_dic_11 = zip(present_train_index_11, present_train_names_11)
present_train_dic_12 = zip(present_train_index_12, present_train_names_12)
present_train_dic_reverse = zip(
    present_train_index_reverse, present_train_names_reverse)
present_train_dic_reverse8 = zip(
    present_train_index_reverse8, present_train_names_reverse8)
present_train_dic_reverse9 = zip(
    present_train_index_reverse9, present_train_names_reverse9)
present_train_dic_reverse11 = zip(
    present_train_index_reverse11, present_train_names_reverse11)
present_train_dic_reverse12 = zip(
    present_train_index_reverse12, present_train_names_reverse12)

pd.DataFrame(absent_train_dic).to_csv(
    index_path+"\absent_train_disc.csv", index=False, header=False)
pd.DataFrame(absent_test_dic).to_csv(
    index_path+"\absent_test_dic.csv", index=False, header=False)
pd.DataFrame(present_train_dic).to_csv(
    index_path+"\present_train_dic.csv", index=False, header=False)
pd.DataFrame(present_test_dic).to_csv(
    index_path+"\present_test_dic.csv", index=False, header=False)
pd.DataFrame(present_train_dic_8).to_csv(
    index_path+"\present_train_dic_8.csv", index=False, header=False)
pd.DataFrame(present_train_dic_9).to_csv(
    index_path+"\present_train_dic_9.csv", index=False, header=False)
pd.DataFrame(present_train_dic_11).to_csv(
    index_path+"\present_train_dic_11.csv", index=False, header=False)
pd.DataFrame(present_train_dic_12).to_csv(
    index_path+"\present_train_dic_12.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse).to_csv(
    index_path+"\present_train_dic_reverse.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse8).to_csv(
    index_path+"\present_train_dic_reverse8.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse9).to_csv(
    index_path+"\present_train_dic_reverse9.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse11).to_csv(
    index_path+"\present_train_dic_reverse11.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse12).to_csv(
    index_path+"\present_train_dic_reverse12.csv", index=False, header=False)

# 保存字典数据
# np.savez(npz_path + r"\absent_train_dic.npz", absent_train_dic)
# np.savez(npz_path + r"\absent_test_dic.npz", absent_train_dic)
# np.savez(npz_path + r"\present_train_dic.npz", present_train_dic)
# np.savez(npz_path + r"\present_test_dic.npz", present_test_dic)
# np.savez(npz_path + r"\present_train_8_dic.npz", present_train_dic_8)
# np.savez(npz_path + r"\present_train_9_dic.npz", present_train_dic_9)
# np.savez(npz_path + r"\present_train_11_dic.npz", present_train_dic_11)
# np.savez(npz_path + r"\present_train_12_dic.npz", present_train_dic_12)
# np.savez(npz_path + r"\present_train_reverse_dic.npz",
#         present_train_dic_reverse)
