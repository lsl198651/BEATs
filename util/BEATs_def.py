# from pickle import TRUE
# from re import split
# from sympy import true
import torch
import sys
import torch.nn as nn
# import torch.optim as optim
# import torchvision
import torch
import torch.nn.functional as F
# from utils import _log_api_usage_once
import matplotlib.pyplot as plt
import csv
import os
import librosa
import torch
# import pywt
import shutil
import pandas as pd
import numpy as np
import logging
import datetime
# import random
# import torchaudio
# from torchaudio import transforms
from IPython.display import display
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from datetime import datetime
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from pydub import AudioSegment
from scipy import signal
# from sklearn import preprocessing
# from torcheval.metrics.functional import binary_auprc, binary_auroc, binary_f1_score, binary_confusion_matrix, binary_accuracy, binary_precision, binary_recall
import warnings
from scipy import signal
# from python_speech_features import logfbank
# from spafe.features.gfcc import erb_spectrogram
# from spafe.utils.preprocessing import SlidingWindow
from util.get_wide_feature import hand_fea
warnings.filterwarnings('ignore')


# def myDownSample_2d(data, sample_Fs, targetFreq):
#     step = sample_Fs // targetFreq
#     newData = np.array([data[:, i] for i in range(
#         data.shape[1]) if i % step == 0]).transpose([1, 0])
#     return newData


# def Mel_Time_Frequency_Spectrum_2(signal, Fs):
#     EPS = 1E-6
#     coef, freqs = pywt.cwt(signal, np.arange(1, 17), 'cgau3')
#     coef = np.abs(coef)
#     coef = myDownSample_2d(coef, Fs, Fs/8)
#     lms = np.log(coef + EPS)

#     # plt.imshow(coef[:,500:800], origin='lower')
#     # plt.ylabel("Wavelet D=16")
#     # plt.xlabel("Time")
#     # # plt.axis('off')
#     # plt.show()

#     return lms


# def Log_GF(wavform):
#     # x, fs = librosa.load(filename, sr=16000)
#     # energy = Energy(x,fs,100,50)
#     fbank_feat_all = []
#     for i in range(wavform.shape[0]):
#         gSpec, gfreqs = erb_spectrogram(wavform[i, :].numpy(), fs=16000, pre_emph=0, pre_emph_coeff=0.97, window=SlidingWindow(
#             0.025, 0.0125, "hamming"), nfilts=64, nfft=512, low_freq=25, high_freq=2000)
#         fbank_feat = gSpec.T
#         # fbank_feat = fbank_feat*energy
#         # fbank_feat = np.log(fbank_feat)
#         # if i == 0:
#         fbank_feat_all.append(fbank_feat)
#     fbank_feat_all = np.stack(fbank_feat_all, axis=0)
#     # else:
#     #     fbank_feat_all = np.vstack((fbank_feat_all, fbank_feat))
#     # gSpec, gfreqs = erb_spectrogram(wavform[1, :].numpy(), fs=16000, pre_emph=0, pre_emph_coeff=0.97, window=SlidingWindow(
#     #     0.025, 0.0125, "hamming"), nfilts=64, nfft=512, low_freq=25, high_freq=2000)
#     # fbank_feat = gSpec.T
#     # # fbank_feat = fbank_feat*energy
#     # fbank_feat = np.log(fbank_feat)
#     return torch.FloatTensor(fbank_feat_all)


def butterworth_low_pass_filter(original_signal, order=3, lowcut=25, highcut=800, sampling_frequency=16000):

    # Get the butterworth filter coefficients
    low = 2*lowcut / sampling_frequency
    high = highcut / sampling_frequency
    B, A = signal.butter(order, [low, high],  btype='bandpass')
    # Forward-backward filter the original signal using the butterworth
    # coefficients, ensuring zero phase distortion
    filtered_signal = np.zeros(original_signal.shape)
    original_signal = original_signal.numpy()
    for i in range(original_signal.shape[0]):
        filtered_signal[i] = signal.filtfilt(B, A, original_signal[i])
    return torch.Tensor(original_signal)

    # band_pass_filtered_signal = signal.filtfilt(B, A, original_signal)
    # return band_pass_filtered_signal


def mkdir(path):
    folder = os.path.exists(path)
    # judge wether make dir or not
    if not folder:
        os.makedirs(path)


def csv_reader_cl(file_name, clo_num):
    """read csv file by column
    """
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


#
def csv_reader_row(file_name, row_num):
    """read the csv row_num-th row"""
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        row = list(reader)
    return row[row_num]


def copy_wav(folder, idlist, mur, traintest):
    """将wav文件夹复制到指定路径"""
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
    """读csv文件"""
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        id = [row[0] for row in reader]  # weight 同列的数据
    return id


def wav_normalize(data):
    """归一化"""
    range = np.max(data) - np.min(data)
    data = (data-np.mean(data))/range
    return data
    # recording -= recording.mean()
    # recording /= recording.abs().max()


def wav_reverse(dir_path, save_path):
    """倒放"""
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            # 读取文件
            temp = AudioSegment.from_file(wav_path, format="wav")
            backplay = temp.reverse()
            # 存为相关格式倒放文件
            reverse_name = subfile.split(".")[0]+"_reverse"
            backplay.export(save_path+reverse_name+".wav", format="wav")


def get_wav_data(dir_path, num=0):
    """返回数据文件"""
    wav = []
    label = []
    file_names = []
    wav_nums = []
    feat = []
    data_length = 6000
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            wav_path = os.path.join(root, subfile)
            if os.path.exists(wav_path):
                # 序号
                num = num+1
                file_names.append(subfile)
                wav_nums.append(num)
                # 数据读取
                print("reading: " + subfile)
                y, sr = librosa.load(wav_path, sr=4000)
                # TODO采样率:4k
                # y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
                y_16k_norm = wav_normalize(y)  # 归一化
                print("num is "+str(num), "y_16k size: "+str(y_16k_norm.size))
                if y_16k_norm.shape[0] < data_length:
                    y_16k_norm = np.pad(
                        y_16k_norm,
                        ((0, data_length - y_16k_norm.shape[0])),
                        "constant",
                        constant_values=(0, 0),
                    )
                elif y_16k_norm.shape[0] > data_length:
                    y_16k_norm = y_16k_norm[-data_length:]
                wav.append(y_16k_norm)
                file_name = subfile.split("_")
                # 标签读取
                if file_name[4] == "Absent":  # Absent
                    label.append(0)
                elif file_name[4] == "Present":  # Present
                    label.append(1)  # 说明该听诊区有杂音
                feat.append(file_name[-1])

    return wav, label, file_names, wav_nums, num, feat


def cal_len(dir_path, csv_path, Murmur: str, id_data, Murmur_locations):
    """计算音频长度"""
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
                waveform_16k = librosa.resample(
                    y=waveform, orig_sr=sr, target_sr=16000)
                print("waveform_16k size: " + str(waveform_16k.size))

                if subfile.split("_")[2] == "Systolic":
                    slen.append(waveform_16k.size)
                else:
                    dlen.append(waveform_16k.size)
    return np.array(slen), np.array(dlen)


def get_mel_features(dir_path, absent_id, present_id):
    """读取csv文件返回feature和label"""
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


class MyDataset(Dataset):
    """my dataset."""

    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata):
        # 直接传递data和label
        # self.len = wavlen
        self.data = torch.tensor(wavdata)
        self.label = torch.tensor(wavlabel)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = self.data[index]
        labelitem = self.label[index]
        return dataitem.float(), labelitem.float()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class DatasetClass(Dataset):
    """继承Dataset类，重写__getitem__和__len__方法
    添加get_idx方法，返回id
    input: wavlabel, wavdata, wavidx

    """

    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata, wavidx, wavebd):
        # 直接传递data和label
        # self.len = wavlen
        embeds = []
        for embed in wavebd:
            embed = int(embed.split('.')[0])
            embeds.append(embed)
        self.wavebd = embeds
        self.data = torch.from_numpy(wavdata)
        self.label = torch.from_numpy(wavlabel)
        self.id = torch.from_numpy(wavidx)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = self.data[index]
        labelitem = self.label[index]
        iditem = self.id[index]
        embeding = self.wavebd[index]
        wide_feat = hand_fea((dataitem, 16000))
        return dataitem.float(), labelitem, iditem, wide_feat, embeding

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)

    # def get_idx(self, index):
    #     iditem = self.id[index]
    #     return iditem


def get_segment_target_list(test_fold, set_type):
    """ get segment target list
        根据csv文件生成并返回segment_target_list
        列表包含所有present的id和对应的位置
    """
    for k in test_fold:
        absent_test_id_path = fr"D:\Shilong\murmur\01_dataset" + \
            set_type+fr"\absent_fold_{k}.csv"
        present_test_id_path = fr"D:\Shilong\murmur\01_dataset" + \
            set_type+fr"\present_fold_{k}.csv"
        absent_test_id = csv_reader_cl(absent_test_id_path, 0)
        present_test_id = csv_reader_cl(present_test_id_path, 0)
    csv_path = r"D:\Shilong\murmur\dataset_all\training_data.csv"
    # get dataset tag from table
    row_line = csv_reader_row(csv_path, 0)
    tag_list = []
    # get index for 'Patient ID' and 'Outcome'
    tag_list.append(row_line.index("Patient ID"))
    tag_list.append(row_line.index("Murmur"))
    tag_list.append(row_line.index("Murmur locations"))
    tag_list.append(row_line.index("Recording locations:"))
    id_data = csv_reader_cl(csv_path, tag_list[0])
    Murmur = csv_reader_cl(csv_path, tag_list[1])
    Murmur_locations = csv_reader_cl(csv_path, tag_list[2])
    Recording_locations = csv_reader_cl(csv_path, tag_list[3])
    # 测试集中所有的id
    test_id = absent_test_id+present_test_id
    # 创建一个空列表segment_present，用来存储有杂音的音频的id和位置
    segment_present = []
    # print(absent_test_id)
    for id in present_test_id:
        # 查此id的murmur状态和听诊位置
        murmurs = Murmur[id_data.index(id)]
        murmur_locations = Murmur_locations[id_data.index(id)]
        if murmurs == 'Present':
            locations = murmur_locations.split('+')
            for loc in locations:
                if loc in ['AV', 'PV', 'TV', 'MV']:
                    # 以id_loc的形式存储present的id和位置
                    segment_present.append(id+'_'+loc)
                else:
                    print(f'[WANGING 1]: {id}_{loc} not in locations')
        else:
            print(f'[WANGING 2]: {id} murmurs is not present?')

    patient_dic = {}
    present_id_loc = {}
    # 创建一个空字典，用来存储id和对应的听诊区,formate: id:听诊区
    # present_id_loc用于保存有杂音的id和对应的听诊区：id：AV+PV+TV
    # print(absent_test_id)
    for id in test_id:
        patient_dic[id] = Recording_locations[id_data.index(id)]
        present_id_loc[id] = Murmur_locations[id_data.index(id)]
    return segment_present, patient_dic, absent_test_id, present_test_id  # , present_id_loc


def segment_classifier(result_list_1=[], test_fold=[], set_type=None):
    """本fn计算了针对每个location和patient的acc和cm
    Args:
        result_list_1 (list, optional): 此列表用来存储分类结果为1对应的id.从test结果中生成传入.
        segment_present (list, optional): _description_. 这是有杂音（=1）的音频target列表，在列表中对应为1，不在则对应为0.
    Returns:
        _type_: _description_
    """
    npy_path_padded = r"D:\Shilong\murmur\01_dataset" + \
        set_type+r"\npyFile_padded\npy_files01"
    for k in test_fold:
        absent_test_index = np.load(
            npy_path_padded + f"\\absent_index_norm01_fold{k}.npy", allow_pickle=True)
        present_test_index = np.load(
            npy_path_padded + f"\\present_index_norm01_fold{k}.npy", allow_pickle=True)
        absent_test_names = np.load(
            npy_path_padded + f"\\absent_name_norm01_fold{k}.npy", allow_pickle=True)
        present_test_names = np.load(
            npy_path_padded + f"\\present_name_norm01_fold{k}.npy", allow_pickle=True)
    """可以测一下这个字典names,index组合对不对"""
    absent_test_dic = dict(zip(absent_test_names, absent_test_index))
    present_test_dic = dict(zip(present_test_names, present_test_index))
    # 所有测试数据的字典
    test_dic = {**absent_test_dic, **present_test_dic}
    # 创建id_pos:idx的字典
    # ------------------------------------------------------------ #
    # -------------------/ segment classifier /------------------- #
    # ------------------------------------------------------------ #
    id_idx_dic = {}
    # 遍历test_dic，生成id_pos:idx的字典
    for file_name, data_index in test_dic.items():
        id_pos = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        # 如果id_pos不在字典中，就创建一个新的键值对
        if not id_pos in id_idx_dic.keys():
            id_idx_dic[id_pos] = [data_index]
        # 如果id_pos在字典中，就把value添加到对应的键值对的值中
        else:
            id_idx_dic[id_pos].append(data_index)
    # id_idx_dic格式：12345_AV: [001,002,003,004,005]

    # 创建一个空字典，用来存储分类结果,formate: id_pos: result
    result_dic = {}
    # 这样就生成了每个听诊区对应的数据索引，然后就可以根据索引读取数据了
    for id_pos, data_index in id_idx_dic.items():
        # 创建空列表用于保存数据索引对应的值
        value_list = []
        # 遍历这个id_pos对应的所有数据索引
        for idx in data_index:
            # 根据索引读取数据
            if idx in result_list_1:
                value_list.append(1)
            else:
                value_list.append(0)
        # 计算平均值作为每一段的最终分类结果，大于0.5就是1，小于0.5就是0,返回字典
        result_dic[id_pos] = np.mean(value_list)
    # result_dic格式：12345_AV: 0.5, 12345_MV: 0.3
    # 获取segment_target_list,这是csv里面读取的有杂音的音频的id和位置
    segment_present, patient_dic, absent_test_id, present_test_id = get_segment_target_list(
        test_fold, set_type)
    # 创建两个列表，分别保存outcome和target列表
    segment_output = []
    segment_target = []
    # 最后，根据target_list，将分类结果转换为0和1并产生outcome_list
    for id_loc, result_value in result_dic.items():
        # TODO: 这里的阈值是0.4,会提升性能吗？
        if result_value >= 0.5:
            segment_output.append(1)
            result_dic[id_loc] = 1
        else:
            segment_output.append(0)
            result_dic[id_loc] = 0
        # 这里计算segment的target
        if id_loc in segment_present:
            segment_target.append(1)
        else:
            segment_target.append(0)
    # 这里需要检查一下segment_target_list和segment_output_list的长度是否一致
    # 计算准确率的代码是这样的吗？？？
    segment_acc = (np.array(segment_output) == np.array(
        segment_target)).sum()/len(segment_target)
    # 计算混淆矩阵
    segment_cm = confusion_matrix(
        segment_target, segment_output)
    # -------------------------------------------------------- #
    # -----------------/ patient classifier /----------------- #
    # -------------------------------------------------------- #
    # patient_result_dic用于保存每个患者每个听诊区的分类结果，formate: id: location1_result,location2_result
    """# ------------------修复bug----------------
    # # 以下五个列表的听诊区数据有重复
    # patient_dic['50115'] = 'AV+MV'
    # patient_dic['49748'] = 'TV+MV'
    # patient_dic['50802'] = 'PV+MV'
    # patient_dic['50782'] = 'PV+TV'
    # patient_dic['49952'] = 'AV+PV+TV'
    # ------------------修复bug---------------"""
    patient_result_dic = {}
    # patient_dic，formate: id:听诊区，123：AV+MV， 456：PV+TV
    for patient_id, locations in patient_dic.items():
        locations = locations.split('+')
        for location in np.unique(locations):
            # 这里除去了phc，因为phc不在result_dic中
            if location in ['AV', 'PV', 'TV', 'MV']:
                id_location = patient_id+'_'+location

            if id_location in result_dic.keys():
                if not patient_id in patient_result_dic.keys():
                    patient_result_dic[patient_id] = result_dic[id_location]
                else:
                    patient_result_dic[patient_id] += result_dic[id_location]
            else:
                # patient_result_dic[patient_id] += 0
                # 正常情况不会报这个错，因为result_dic中的id_loc都是在segment_target_list中的
                print('[WANGING 3]: ' + id_location+' not in result_dic')
    # 遍历patient_result_dic，计算每个患者的最终分类结果
    patient_output_dic = {}
    patient_output = []
    patient_target = []
    for patient_id, result in patient_result_dic.items():
        # 做output
        if result == 0:    # 全为0 absent
            patient_output_dic[patient_id] = 0
            patient_output.append(0)
        elif result > 0:  # 不全为0 present
            patient_output_dic[patient_id] = 1
            patient_output.append(1)
        else:
            print('[WANGING 4]: result value is negtive?')  # 有负数
        # 做target
        if patient_id in absent_test_id:
            patient_target.append(0)
        elif patient_id in present_test_id:
            patient_target.append(1)
        else:
            print('[WANGING 5]: '+patient_id+' not in test_id')
    # 统计patient的错误id
    patient_id_test = list(patient_output_dic.keys())
    # 保存错误的id
    patient_error_id = []
    for patient_index in range(len(patient_output)):
        if patient_output[patient_index] != patient_target[patient_index]:
            patient_error_id.append(patient_id_test[patient_index])
    print(patient_error_id)

    # # 计算准确率和混淆矩阵
    # test_input, test_target)
    # # 计算准确率
    # #  patient_output.eq(patient_target).sum().item()
    # patient_acc = (np.array(patient_output) == np.array(
    #     patient_target)).sum()/len(patient_target)
    # # 计算混淆矩阵
    # patient_cm = confusion_matrix(
    #     patient_target, patient_output)
    return segment_acc, segment_cm, patient_output, patient_target, patient_error_id


class BCEFocalLoss(nn.Module):
    """BiFocal Loss"""

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        self.gamma = self.gamma.view(target.size)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
            1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss_VGG(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True, weight=None):
        super(FocalLoss_VGG, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight
        # if isinstance(alpha, list):
        #     assert len(alpha) == num_classes
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha < 1
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1 - alpha)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                torch.argmax(inputs, dim=1).float(), targets, weight=self.weight, reduce=False)
        else:
            CE_loss = nn.CrossEntropyLoss(
                inputs, targets, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.log(torch.softmax(input, dim=1))  #
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.cross_entropy(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    loss = loss.mean()
    return loss


def logger_init(
    log_level=logging.DEBUG,
    log_dir=r"./log",
):
    """初始化logging"""
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 指定日志格式
    date = datetime.now()
    log_path = os.path.join(
        log_dir, str(datetime.now().strftime("%Y-%m%d %H%M")) + ".log"
    )
    formatter = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=formatter,
        datefmt="%m%d %H%M",
        handlers=[logging.FileHandler(
            log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.disable(logging.DEBUG)


class save_info(object):
    """设置logging格式"""

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


def draw_confusion_matrix(
    cm,
    label_name,
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
    row_sums = np.sum(cm, axis=1)  # 计算每行的和
    cm = cm.T
    plt.imshow(cm.T, cmap="Reds")
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name)

    plt.tight_layout()
    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            # color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            # value = float(format("%.4f" % cm[i, j]))
            str_value = "{}".format(cm[i, j])
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
