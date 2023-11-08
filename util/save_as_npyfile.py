import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from BEATs_def import get_wav_data, get_patientid
from dataAugmentation import data_Auge
import pandas as pd


# ========================/ parameteres define /========================== #
# murmur_positoin = ["_AV", "_MV", "_PV", "_TV"]
# murmur_ap = ["Absent\\", "Present\\"]
# period = ["Systolic", "Diastolic"]


# ========================/ Data Augementation /========================== #
"""数据增强，包括时间拉伸和反转"""
root_path = r"D:\Shilong\murmur\01_dataset\06_new5fold"
npy_path_padded = root_path+r"\npyFile_padded\npy_files01"
index_path = root_path + r"\npyFile_padded\index_files01"
if not os.path.exists(npy_path_padded):
    os.makedirs(npy_path_padded)
if not os.path.exists(index_path):
    os.makedirs(index_path)
for k in range(5):
    src_fold_root_path = root_path+r"\fold_set_"+str(k)
    data_Auge(src_fold_root_path)
    for folder in os.listdir(src_fold_root_path):
        dataset_path = os.path.join(src_fold_root_path, folder)
        if k == 0 and folder == "absent":
            features, label, names, index, data_id = get_wav_data(
                dataset_path, num=0)  # absent
        else:
            features, label, names, index, data_id = get_wav_data(
                dataset_path, data_id)  # absent
        np.save(npy_path_padded +
                f"\\{folder}_features_norm01_fold{k}.npy", features)
        np.save(npy_path_padded +
                f"\\{folder}_labels_norm01_fold{k}.npy", label)
        np.save(npy_path_padded +
                f"\\{folder}_index_norm01_fold{k}.npy", index)
        np.save(npy_path_padded +
                f"\\{folder}_name_norm01_fold{k}.npy", names)
        absent_train_dic = zip(index, names)
        pd.DataFrame(absent_train_dic).to_csv(
            index_path+f"\\fold{k}_{folder}_disc.csv", index=False, header=False)

# ========================/ file path /========================== #
# get absent / present patient_id
# csv_folder = r"D:\Shilong\murmur\03_circor_statest"
# id_data_path = csv_folder+r"\id_data.csv"
# absent_csv_path = csv_folder+r"\absent_id.csv"
# present_csv_path = csv_folder+r"\present_id.csv"
# Diastolic_murmur_timing_path = (
#     csv_folder+r"\Diastolic_murmur_timing.csv"
# )
# Systolic_murmur_timing_path = (
#     csv_folder+r"\Systolic_murmur_timing.csv"
# )
# Murmur_locations_path = csv_folder+r"\Murmur_locations.csv"

wav_filepath = root_path+r"\fold_set_"+str(k)
absent_train_path = wav_filepath+r"\absent"
absent_test_path = wav_filepath+r"\absent"
present_train_path = wav_filepath+r"\present"
present_test_path = wav_filepath+r"\present"
present_train_path_8 = wav_filepath+r"\time_stretch0.8"
present_train_path_9 = wav_filepath+r"\time_stretch0.9"
present_train_path_11 = wav_filepath+r"\time_stretch1.1"
present_train_path_12 = wav_filepath+r"\time_stretch1.2"
present_train_path_reverse = wav_filepath+r"\reverse"
present_train_path_reverse8 = wav_filepath+r"\reverse0.8"
present_train_path_reverse9 = wav_filepath+r"\reverse0.9"
present_train_path_reverse11 = wav_filepath+r"\reverse1.1"
present_train_path_reverse12 = wav_filepath+r"\reverse1.2"


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


np.save(npy_path_padded + r"\absent_train_features_norm.npy",
        absent_train_features)
np.save(npy_path_padded + r"\absent_test_features_norm.npy",
        absent_test_features)
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

index_path = r"D:\Shilong\murmur\01_dataset\05_5fold\npyFile_padded\normalized\index_files"
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
    index_path+r"\absent_train_disc.csv", index=False, header=False)
pd.DataFrame(absent_test_dic).to_csv(
    index_path+r"\absent_test_dic.csv", index=False, header=False)
pd.DataFrame(present_train_dic).to_csv(
    index_path+r"\present_train_dic.csv", index=False, header=False)
pd.DataFrame(present_test_dic).to_csv(
    index_path+r"\present_test_dic.csv", index=False, header=False)
pd.DataFrame(present_train_dic_8).to_csv(
    index_path+r"\present_train_dic_8.csv", index=False, header=False)
pd.DataFrame(present_train_dic_9).to_csv(
    index_path+r"\present_train_dic_9.csv", index=False, header=False)
pd.DataFrame(present_train_dic_11).to_csv(
    index_path+r"\present_train_dic_11.csv", index=False, header=False)
pd.DataFrame(present_train_dic_12).to_csv(
    index_path+r"\present_train_dic_12.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse).to_csv(
    index_path+r"\present_train_dic_reverse.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse8).to_csv(
    index_path+r"\present_train_dic_reverse8.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse9).to_csv(
    index_path+r"\present_train_dic_reverse9.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse11).to_csv(
    index_path+r"\present_train_dic_reverse11.csv", index=False, header=False)
pd.DataFrame(present_train_dic_reverse12).to_csv(
    index_path+r"\present_train_dic_reverse12.csv", index=False, header=False)

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
