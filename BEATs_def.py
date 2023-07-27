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
def get_wav_data(dir_path, csv_path):
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

                file_name = subfile.split("_")
                # 标签读取
                if file_name[4] == "Absent":  # Absent
                    label.append(0)
                if file_name[4] == "Present":  # Present
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
            if os.path.exists(wav_path):
                # 数据读取
                print("reading: " + subfile)
                waveform, sr = librosa.load(wav_path)
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
    log_dir=r"./ResultFile",
):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 指定日志格式
    date = datetime.now()
    log_path = os.path.join(
        log_dir, str(datetime.now().strftime("%Y-%m%d %H%M")) + ".log"
    )
    formatter = "[%(asctime)s - %(levelname)s] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=formatter,
        datefmt="%Y-%m%d %H%M",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.disable(logging.DEBUG)


# ========================/ logging formate /========================== #
class save_info(object):
    def __init__(self, epoch_num, epoch, train_loss, test_acc, test_loss):
        self.epoch = epoch
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss

        logging.info(f"epoch: " + str(self.epoch + 1) + "/" + str(epoch_num))
        logging.info(f"train_loss: " + str("{:.3f}".format(self.train_loss)))
        logging.info(
            f"test_acc: "
            + str("{:.3f}%".format(self.test_acc))
            + ", test_loss: "
            + str("{:.3f}".format(self.test_loss))
        )
        logging.info(f"======================================")


# ========================/ train and test /========================== #
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(
    label_true,
    label_pred,
    label_name,
    normlize,
    title="Confusion Matrix",
    pdf_save_path=None,
    dpi=600,
    epoch=0,
):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param normlize: 是否设元素为百分比形式
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          normlize=True,
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(label_true, label_pred)

    row_sums = np.sum(cm, axis=1)  # 计算每行的和
    cm2 = cm / row_sums[:, np.newaxis]  # 广播计算每个元素占比
    cm2 = cm2.T
    cm = cm.T
    plt.imshow(cm, cmap="Reds")
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()
    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            # color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            # value = float(format("%.4f" % cm[i, j]))
            str_value = "{}({:.2%})".format(cm[i, j], cm2[i, j])
            plt.text(
                i,
                j,
                str_value,
                verticalalignment="center",
                horizontalalignment="center",
                # color=color,
            )

    # plt.show()
    if not pdf_save_path is None:
        if not os.path.exists(pdf_save_path):
            os.makedirs(pdf_save_path)
        plt.savefig(
            pdf_save_path + "/epoch" + str(epoch) + ".png",
            bbox_inches="tight",
            dpi=dpi,
        )
        plt.close()
