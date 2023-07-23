# Author: Shilong
# Creat Data: 2023-6-16
# Modify Date:2023-
# Description: load fin-tuned BEATs model
"""
Load Pre-Trained Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from BEATs import BEATs_Pre_Train_itere3
from BEATs_def import (
    get_patientid,
    get_wav_data,
    copy_wav,
    get_mel_features,
    csv_reader_cl,
    MyDataset,
    logger_init,
    save_info,
    cal_len,
)

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

absent_train_csv_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
absent_test_csv_path = r"D:\Shilong\murmur\03_circor_states\test_csv"
present_train_csv_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
present_test_csv_path = r"D:\Shilong\murmur\03_circor_states\test_csv"

filepath = r"D:\Shilong\murmur\03_circor_states"
absent_train_path = r"D:\Shilong\murmur\03_circor_states\train\Absent"
absent_test_path = r"D:\Shilong\murmur\03_circor_states\test\Absent"
present_train_path = r"D:\Shilong\murmur\03_circor_states\train\Present"
present_test_path = r"D:\Shilong\murmur\03_circor_states\test\Present"

folder = r"D:\Shilong\murmur\03_circor_statest"
npy_path = r"D:\Shilong\murmur\03_circor_states\npyFile"
npy_path_padded = r"D:\Shilong\murmur\03_circor_states\npyFile_padded"

path = r"D:\Shilong\murmur\03_circor_states\csv"
train_path = r"D:\Shilong\murmur\03_circor_states\train_csv"
test_path = r"D:\Shilong\murmur\03_circor_states\test_csv"
# ========================/ get lists /========================== #
id_data = get_patientid(id_data_path)
absent_patient_id = get_patientid(absent_csv_path)
present_patient_id = get_patientid(present_csv_path)
Diastolic_murmur_timing = get_patientid(Diastolic_murmur_timing_path)
Systolic_murmur_timing = get_patientid(Systolic_murmur_timing_path)
Murmur_locations = get_patientid(Murmur_locations_path)

"""# ========================/ get wav data, length=10000 /========================== # 
absent_train_features,absent_train_label = get_wav_data(absent_train_path,absent_train_csv_path,'Absent',id_data,Murmur_locations)# absent
absent_test_features,absent_test_label = get_wav_data(absent_test_path,absent_test_csv_path,'Absent',id_data,Murmur_locations)# absent
present_train_features,present_train_label= get_wav_data(present_train_path,present_train_csv_path,'Present',id_data,Murmur_locations)# present
present_test_features,present_test_label=get_wav_data(present_test_path,present_test_csv_path,'Present',id_data,Murmur_locations)# present

# # # # ========================/ save as npy file /========================== # 
np.save(npy_path_padded+r'\absent_train_features.npy',absent_train_features)
np.save(npy_path_padded+r'\absent_test_features.npy',absent_test_features)
np.save(npy_path_padded+r'\present_train_features.npy',present_train_features)
np.save(npy_path_padded+r'\present_test_features.npy',present_test_features)

np.save(npy_path_padded+r'\absent_train_label.npy',absent_train_label)
np.save(npy_path_padded+r'\absent_test_label.npy',absent_test_label)
np.save(npy_path_padded+r'\present_train_label.npy',present_train_label)
np.save(npy_path_padded+r'\present_test_label.npy',present_test_label)
"""
# ========================/ load npy file /========================== #
# absent_train_features = np.load(npy_path+r'\absent_train_features.npy',allow_pickle=True)
# absent_test_features = np.load(npy_path+r'\absent_test_features.npy',allow_pickle=True)
# present_train_features = np.load(npy_path+r'\present_train_features.npy',allow_pickle=True)
# present_test_features = np.load(npy_path+r'\present_test_features.npy',allow_pickle=True)

# absent_train_label = np.load(npy_path+r'\absent_train_label.npy',allow_pickle=True)
# absent_test_label = np.load(npy_path+r'\absent_test_label.npy',allow_pickle=True)
# present_train_label = np.load(npy_path+r'\present_train_label.npy',allow_pickle=True)
# present_test_label = np.load(npy_path+r'\present_test_label.npy',allow_pickle=True)

# ========================/ load npy padded file /========================== #
absent_train_features = np.load(
    npy_path_padded + r"\absent_train_features.npy", allow_pickle=True
)
absent_test_features = np.load(
    npy_path_padded + r"\absent_test_features.npy", allow_pickle=True
)
present_train_features = np.load(
    npy_path_padded + r"\present_train_features.npy", allow_pickle=True
)
present_test_features = np.load(
    npy_path_padded + r"\present_test_features.npy", allow_pickle=True
)

absent_train_label = np.load(
    npy_path_padded + r"\absent_train_label.npy", allow_pickle=True
)
absent_test_label = np.load(
    npy_path_padded + r"\absent_test_label.npy", allow_pickle=True
)
present_train_label = np.load(
    npy_path_padded + r"\present_train_label.npy", allow_pickle=True
)
present_test_label = np.load(
    npy_path_padded + r"\present_test_label.npy", allow_pickle=True
)

# ========================/ get features & labels /========================== #
# test_features,test_label=get_mel_features(path,absent_patient_id,present_patient_id)
"""train_features,train_label=get_mel_features(train_path,absent_patient_id,present_patient_id)
test_features,test_label=get_mel_features(test_path,absent_patient_id,present_patient_id)
train_features=train_features.astype(float)
train_label=train_label.astype(float)
test_features=test_features.astype(float)
test_label=test_label.astype(float)"""

# ========================/ label encoder /========================== #
train_label = np.hstack((absent_train_label, present_train_label))
test_label = np.hstack((absent_test_label, present_test_label))
train_features = np.vstack((absent_train_features, present_train_features))
test_features = np.vstack((absent_test_features, present_test_features))

# ========================/ train test /========================== #
train_features = train_features.astype(float)
train_label = train_label.astype(float)
test_features = test_features.astype(float)
test_label = test_label.astype(float)

# ========================/ MyDataset /========================== #
train_set = MyDataset(wavlabel=train_label, wavdata=train_features)
test_set = MyDataset(wavlabel=test_label, wavdata=test_features)

# ========================/ HyperParameters /========================== #
batch_size = 128
learning_rate = 0.0005
num_epochs = 200
padding_size = train_features.shape[1]  # 3500
padding = torch.zeros(
    batch_size, padding_size
).bool()  # we randomly mask 75% of the input patches,
padding_mask = torch.Tensor(padding)

# ========================/ dataloader /========================== #
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
print("Dataloader is ok")  # 最后再打印一下新的模型

# ========================/ load model /========================== #
MyModel = BEATs_Pre_Train_itere3()

# ========================/ model add fc-layer /========================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel.to(DEVICE)  # 放到设备中
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [{"params": MyModel.last_layer.parameters()}],
    lr=learning_rate,
    betas=(0.9, 0.98),
)  # 指定 新加的fc层的学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
min_lr = 10.0
max_lr = 0.0

# ========================/ train model /========================== #
# 定义训练函数
def train_model(model, device, train_loader, test_loader, padding, epochs,lr=[]):
    # train model
    model.train()
    for data_t, label_t in train_loader:
        data_t, label_t = data_t.to(device), label_t.to(device)
        padding = padding.to(device)
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        predict = model(data_t, padding)
        loss = criterion(predict, label_t.long())
        loss.backward()
        optimizer.step()

    train_loss = loss.item()

    # evaluate model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_v, label_v in test_loader:
            data_v, label_v, padding = (
                data_v.to(device),
                label_v.to(device),
                padding.to(device),
            )
            optimizer.zero_grad()
            predict_v = model(data_v, padding)
            # recall = recall_score(y_hat, y)
            test_loss += criterion(
                predict_v, label_v.long()
            ).item()  # sum up batch loss
            pred = predict_v.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(label_v.view_as(pred)).sum().item()
    scheduler.step()

    for group in optimizer.param_groups:
        lr_now = group["lr"]
    lr.append(lr_now)

    # 更新权值
    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_set)

    max_test_acc=0.
    max_test_acc = max(max_test_acc, correct / len(test_set))

    tb_writer.add_scalar("train_loss", train_loss, epochs)
    tb_writer.add_scalar("test_loss", test_loss, epochs)
    tb_writer.add_scalar("test_acc", test_acc, epochs)
    tb_writer.add_scalar("learning_rate", lr_now, epochs)
    # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
    logging.info(f"epoch: " + str(epochs + 1) + "/" + str(num_epochs))
    logging.info(f"train_loss: " + str("{:.4f}".format(train_loss)))
    logging.info(f"learning_rate: " + str("{:.4f}".format(lr_now)))
    logging.info(
        f"test_acc: "
        + str("{:.4f}%".format(test_acc))
        + ", test_loss: "
        + str("{:.4f}".format(test_loss))
    )
    logging.info(
        f"max_test_acc: "
        + str("{:.4f}".format(max_test_acc))
    )
    logging.info(
        f"max_lr: "
        + str("{:.4f}".format(max(lr)))
        + ", min_lr: "
        + str("{:.4f}".format(min(lr)))
    )
    logging.info(f"======================================")

# ========================/ training and logging info /========================== #
logger_init()
logging.info("# batch_size = " + str(batch_size))
logging.info("# learning_rate = " + str(learning_rate))
logging.info("# num_epochs = " + str(num_epochs))
logging.info("# padding_size = " + str(padding_size))
logging.info("# criterion = " + str(criterion))
logging.info("# scheduler = " + str(scheduler))
logging.info("# optimizer = " + str(optimizer))
logging.info("-------------------------------")
tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now())[:13])

for epoch in range(num_epochs):
    train_model(
        model=MyModel,
        device=DEVICE,
        train_loader=train_loader,
        test_loader=test_loader,
        padding=padding_mask,
        epochs=epoch,
    )
tb_writer.close()
