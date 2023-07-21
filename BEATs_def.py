import csv
import os
import librosa
import torch, shutil
import pandas as pd
import numpy as np
import logging
import datetime
import torchaudio
from datetime import datetime
import sys
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def csv_reader_cl(file_name, clo_num):
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


def copy_wav(folder, idlist, mur, traintest):
    for patient_id in idlist:
        dir_path = folder + "\\" + mur + "\\" + patient_id
        # print(dir_path)
        for root, dir, file in os.walk(dir_path):
            for subdir in dir:
                subdir_path = os.path.join(root, subdir)
                print(subdir_path)
                # if os.path.exists(dir_path):
                shutil.copytree(subdir_path, traintest + "\\" + subdir)
    # idlist.to_scv(folder+"\\"+traintest+"\\"+mur+traintest+".csv", index=False, header=False)


def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        id = [row[0] for row in reader]  # weight 同列的数据
    return id


# 读取数据并打标签
def get_wav_data(dir_path, csv_path, Murmur: str, id_data, Murmur_locations):
    wav = []
    label = []
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 数据读取
                print("reading: " + subfile)
                y, sr = librosa.load(wav_path)
                y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                # print("y_16k size: "+y_16k.size)
                if y_16k.shape[0] < 3500:
                    y_16k = np.pad(
                        y_16k,
                        (0, 3500 - y_16k.shape[0]),
                        "constant",
                        constant_values=(0, 0),
                    )
                elif y_16k.shape[0] > 3500:
                    y_16k = y_16k[0:3500]
                wav.append(y_16k)
                # feature = pd.DataFrame(y_16k)
                # save_path = csv_path +"\\"+ subfile+ ".csv"
                # print("shape: "+feature.shape())
                # feature.to_csv(save_path, index=False, header=False)

                # 标签读取
                if Murmur == "Absent":  # Absent
                    label.append(0)
                else:  # Present
                    # 先找到id对应的index,再通过索引找到murmur_locations和timing
                    murmur_ap = subfile.split("_")[4]
                    if murmur_ap == "Absent":  # 说明该听诊区有杂音
                        label.append(0)  # 舒张期全部认为没有杂音
                    else:
                        label.append(1)  # 说明该听诊区无杂音
    return np.array(wav), np.array(label)


def cal_len(dir_path, csv_path, Murmur: str, id_data, Murmur_locations):
    slen = []
    dlen = []
    # label=[]
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 数据读取
                print("reading: " + subfile)
                print("reading: " + subfile)
                waveform, sr = librosa.load(wav_path, sr=4000)
                waveform_16k = librosa.resample(y=waveform, orig_sr=sr, target_sr=16000)
                print("waveform_16k size: " + str(waveform_16k.size))
                waveform_16k = librosa.resample(y=waveform, orig_sr=sr, target_sr=16000)
                print("waveform_16k size: " + str(waveform_16k.size))
                if subfile.split("_")[2] == "Systolic":
                    slen.append(waveform_16k.size)
                else:
                    dlen.append(waveform_16k.size)
    return np.array(slen), np.array(dlen)


"""
读取csv文件返回feature和label
"""


def get_mel_features(dir_path, absent_id, present_id):
    feature_list = []
    label_list = []
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            csv_path = os.path.join(root, subfile)
            print("reading: " + subfile)
            csv_path = os.path.join(root, subfile)
            print("reading: " + subfile)
            # filename = r'E:\Shilong\murmur\LM_wav_dataset\train_csv\2530_AV_1.wav.csv'
            df = pd.read_csv(csv_path, header=None)
            data = np.array(df)
            if data.shape[0] < 24:
                data = np.pad(
                    data, (0, 24 - data.shape[0]), "constant", constant_values=(0, 0)
                )
                data = data[:, 0:768]
            elif data.shape[0] > 24:
                data = data[0:24, :]
            feature_list.append(data)
            id = subfile.split("_")[0]
            if id in absent_id:
                label_list.append(1)
            if id in present_id:
                label_list.append(0)

    return np.array(feature_list), np.array(label_list)


# ========================/ dataset Class /========================== #
class MyDataset(Dataset):
    """my dataset."""

    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata):
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
def logger_init(
    log_level=logging.DEBUG,
    log_dir=r"D:\Shilong\murmur\00_Code\LM\BEATs\ResultFile",
):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 指定日志格式
    date = datetime.now()
    log_path = os.path.join(log_dir, str(date)[:13] + "-" + str(date.minute) + ".log")
    formatter = "[%(asctime)s - %(levelname)s:] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=formatter,
        datefmt="%Y-%d-%m %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


# ========================/ logging formate /========================== #
class save_info(object):
    def __init__(self, epoch_num, epoch, train_loss, test_acc, test_loss):
        self.epoch = epoch
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss

        logging.info(f"epoch: " + str(self.epoch+1) + "/" + str(epoch_num))
        logging.info(f"train_loss: " + str("{:.3f}".format(self.train_loss)))
        logging.info(
            f"test_acc: "
            + str("{:.3f}%".format(self.test_acc))
            + ", test_loss: "
            + str("{:.3f}".format(self.test_loss))
        )
        logging.info(f"======================================")

# ========================/ train and test /========================== #