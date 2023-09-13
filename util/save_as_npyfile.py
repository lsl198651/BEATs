import os
import shutil
import librosa
import numpy as np
import csv
from BEATs_def import get_wav_data, get_patientid


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
present_train_path_12 = wav_filepath+r"\trainset\time_stretch1.2"
# ========================/ get lists /========================== #
# id_data = get_patientid(id_data_path)
# absent_patient_id = get_patientid(absent_csv_path)
# present_patient_id = get_patientid(present_csv_path)
# Diastolic_murmur_timing = get_patientid(Diastolic_murmur_timing_path)
# Systolic_murmur_timing = get_patientid(Systolic_murmur_timing_path)
# Murmur_locations = get_patientid(Murmur_locations_path)

npy_path_padded = wav_filepath+r"\npyFile_padded"

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

# # # ========================/ save as npy file /========================== #
# 保存特征数据
if not os.path.exists(npy_path_padded):
    os.makedirs(npy_path_padded)
np.save(npy_path_padded + r"\absent_train_features.npy", absent_train_features)
np.save(npy_path_padded + r"\absent_test_features.npy", absent_test_features)
np.save(npy_path_padded + r"\present_train_features.npy", present_train_features)
np.save(npy_path_padded + r"\present_test_features.npy", present_test_features)
# 保存标签数据
np.save(npy_path_padded + r"\absent_train_label.npy", absent_train_label)
np.save(npy_path_padded + r"\absent_test_label.npy", absent_test_label)
np.save(npy_path_padded + r"\present_train_label.npy", present_train_label)
np.save(npy_path_padded + r"\present_test_label.npy", present_test_label)

# # 保存增强后的特征和标签
# present_train_features_8, present_train_label_8 = get_wav_data(
#     present_train_path_8
# )  # present
# present_train_features_12, present_train_label_12 = get_wav_data(
#     present_train_path_12
# )  # present
# np.save(npy_path_padded + r"\present_train_features_8.npy",
#         present_train_features_8)
# np.save(npy_path_padded + r"\present_train_features_12.npy",
#         present_train_features_12)
# np.save(npy_path_padded + r"\present_train_label_8.npy", present_train_label_8)
# np.save(npy_path_padded + r"\present_train_label_12.npy", present_train_label_12)
