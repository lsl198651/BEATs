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
import logging
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# from torch.utils.data.sampler import WeightedRandomSampSler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
from datetime import datetime
from torch import optim
from transformers import optimization
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
    draw_confusion_matrix,
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
absent_train_path = r"D:\Shilong\murmur\03_circor_states\trainset\absent"
absent_test_path = r"D:\Shilong\murmur\03_circor_states\testset\absent"
present_train_path = r"D:\Shilong\murmur\03_circor_states\trainset\present"
present_test_path = r"D:\Shilong\murmur\03_circor_states\testset\present"

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
absent_train_features,absent_train_label = get_wav_data(absent_train_path,absent_train_csv_path)# absent
absent_test_features,absent_test_label = get_wav_data(absent_test_path,absent_test_csv_path)# absent
present_train_features,present_train_label= get_wav_data(present_train_path,present_train_csv_path)# present
present_test_features,present_test_label=get_wav_data(present_test_path,present_test_csv_path)# present

# # # # ========================/ save as npy file /========================== # 
np.save(npy_path_padded+r'\absent_train_features.npy',absent_train_features)
np.save(npy_path_padded+r'\absent_test_features.npy',absent_test_features)
np.save(npy_path_padded+r'\present_train_features.npy',present_train_features)
np.save(npy_path_padded+r'\present_test_features.npy',present_test_features)

np.save(npy_path_padded+r'\absent_train_label.npy',absent_train_label)
np.save(npy_path_padded+r'\absent_test_label.npy',absent_test_label)
np.save(npy_path_padded+r'\present_train_label.npy',present_train_label)
np.save(npy_path_padded+r'\present_test_label.npy',present_test_label)"""

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
Data_Augmentation = True
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
if Data_Augmentation is True:
    present_train_features_8 = np.load(
        npy_path_padded + r"\present_train_features_8.npy", allow_pickle=True
    )
    present_train_features_12 = np.load(
        npy_path_padded + r"\present_train_features_12.npy", allow_pickle=True
    )
    present_train_label_8 = np.load(
        npy_path_padded + r"\present_train_label_8.npy", allow_pickle=True
    )
    present_train_label_12 = np.load(
        npy_path_padded + r"\present_train_label_12.npy", allow_pickle=True
    )

    absent_size = int(
        (
            present_train_features.shape[0]
            + present_train_features_12.shape[0]
            + present_train_features_8.shape[0]
        )
    )
    List_train = random.sample(range(1, absent_train_features.shape[0]), absent_size)
    absent_train_features = absent_train_features[List_train]
    absent_train_label = absent_train_label[List_train]
    train_label = np.hstack(
        (
            absent_train_label,
            present_train_label,
            present_train_label_8,
            present_train_label_12,
        )
    )

    train_features = np.vstack(
        (
            absent_train_features,
            present_train_features,
            present_train_features_8,
            present_train_features_12,
        )
    )
else:
    absent_size = int(present_train_features.shape[0])
    List_train = random.sample(range(1, absent_train_features.shape[0]), absent_size)
    absent_train_features = absent_train_features[List_train]
    absent_train_label = absent_train_label[List_train]
    train_label = np.hstack(
        (
            absent_train_label,
            present_train_label,
        )
    )

    train_features = np.vstack(
        (
            absent_train_features,
            present_train_features,
        )
    )
# ap_ratio = 1
# absent_size = int(present_train_features.shape[0] * ap_ratio)


# List_train = random.sample(range(1, absent_train_features.shape[0]), absent_size)
# absent_train_features = absent_train_features[List_train]
# absent_train_label = absent_train_label[List_train]
# List_test = random.sample(range(1, absent_test_features.shape[0]), test_absent_size)
# absent_test_features = absent_test_features[List_test]
# absent_test_label = absent_test_label[List_test]

# ========================/ get features & labels /========================== #
# test_features,test_label=get_mel_features(path,absent_patient_id,present_patient_id)
"""
train_features,train_label=get_mel_features(train_path,absent_patient_id,present_patient_id)
test_features,test_label=get_mel_features(test_path,absent_patient_id,present_patient_id)
"""


# ========================/ label encoder /========================== #
# train_label = np.hstack((absent_train_label, present_train_label))
test_label = np.hstack((absent_test_label, present_test_label))
# train_features = np.vstack((absent_train_features, present_train_features))
test_features = np.vstack((absent_test_features, present_test_features))

train_present_size = np.sum(train_label == 1)
train_absent_size = np.sum(train_label == 0)
test_present_size = np.sum(test_label == 1)
test_absent_size = np.sum(test_label == 0)
# ========================/ train test /========================== #
train_features = train_features.astype(float)
train_label = train_label.astype(int)
test_features = test_features.astype(float)
test_label = test_label.astype(int)
trainset_size = train_features.shape[0]
testset_size = test_features.shape[0]
# ========================/ MyDataset /========================== #
train_set = MyDataset(wavlabel=train_label, wavdata=train_features)
test_set = MyDataset(wavlabel=test_label, wavdata=test_features)

# ========================/ HyperParameters /========================== #
batch_size = 128
learning_rate = 0.000001
num_epochs = 1000
loss_type = "CE"
train_total = True
padding_size = train_features.shape[1]  # 3500
padding = torch.zeros(
    batch_size, padding_size
).bool()  # we randomly mask 75% of the input patches,
padding_v = torch.zeros(
    1, padding_size
).bool()  # we randomly mask 75% of the input patches,
padding_mask = torch.Tensor(padding)
# padding_mask_v = torch.Tensor(padding_v)
# ========================/ dataloader /========================== #
# 如果label为1，那么对应的该类别被取出来的概率是另外一个类别的2倍
# weights = [9 if label == 1 else 1 for data, label in train_set]
# Data_sampler = WeightedRandomSampler(weights, num_samples=9, replacement=True)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
print("Dataloader is ready")  # 最后再打印一下新的模型

# ========================/ load model /========================== #
MyModel = BEATs_Pre_Train_itere3()

# ========================/ model add fc-layer /========================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel.to(DEVICE)  # 放到设备中

if loss_type == "BCE":
    loss_fn = nn.BCEWithLogitsLoss()
elif loss_type == "CE":
    loss_fn = nn.CrossEntropyLoss()

if train_total is True:
    optimizer = torch.optim.AdamW(MyModel.parameters(), lr=learning_rate)
else:
    for param in MyModel.BEATs.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, MyModel.parameters()),
        lr=learning_rate,
    )
# 指定 新加的fc层的学习率

# ========================/ setup warmup lr /========================== #
warm_up_ratio = 0.1
total_steps = len(train_loader) * num_epochs

scheduler = None
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = optimization.get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warm_up_ratio * total_steps,
#     num_training_steps=total_steps,
# )


# ========================/ train model /========================== #
# 定义训练函数
def train_model(
    model,
    device,
    train_loader,
    test_loader,
    padding,
    epochs,
    lr=[],
    max_test_acc=[],
    max_train_acc=[],
):
    # train model
    train_loss = 0
    correct_t = 0
    # for amp
    scaler = GradScaler()
    model.train()
    train_len = 0

    for data_t, label_t in train_loader:
        data_t, label_t = data_t.to(device), label_t.to(device)
        padding = padding.to(device)

        # with torch.cuda.amp.autocast():
        predict = model(data_t, padding)
        # with autocast():
        loss = loss_fn(predict, label_t.long())
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_t = predict.max(1, keepdim=True)[1]
        train_len += len(pred_t)
        # get the index of the max log-probability
        correct_t += pred_t.eq(label_t.view_as(pred_t)).sum().item()

    # evaluate model
    model.eval()
    label = []
    pred = []
    test_loss = 0
    correct_v = 0
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
            test_loss += loss_fn(predict_v, label_v.long()).item()  # sum up batch loss
            pred_v = predict_v.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct_v += pred_v.eq(label_v.view_as(pred_v)).sum().item()
            pred.extend(pred_v.cpu().tolist())
            label.extend(label_v.cpu().tolist())
    if scheduler is not None:
        scheduler.step()

    for group in optimizer.param_groups:
        lr_now = group["lr"]
    lr.append(lr_now)

    # 更新权值
    test_loss /= len(pred)
    test_acc = correct_v / len(pred)
    train_loss /= train_len
    train_acc = correct_t / train_len

    max_train_acc.append(train_acc)
    max_test_acc.append(test_acc)
    max_acc_t = max(max_train_acc)
    max_acc_v = max_test_acc[max_train_acc.index(max_acc_t)]

    tb_writer.add_scalar("train_acc", train_acc, epochs)
    tb_writer.add_scalar("test_acc", test_acc, epochs)
    tb_writer.add_scalar("train_loss", train_loss, epochs)
    tb_writer.add_scalar("test_loss", test_loss, epochs)
    tb_writer.add_scalar("learning_rate", lr_now, epochs)

    # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
    logging.info(f"epoch: " + str(epochs + 1) + "/" + str(num_epochs))
    logging.info(f"learning_rate: " + str("{:.6f}".format(lr_now)))
    logging.info(
        f"train_acc: "
        + str("{:.3%}".format(train_acc))
        + ", train_loss: "
        + str("{:.4f}".format(train_loss))
    )
    logging.info(
        f"test_acc: "
        + str("{:.3%}".format(test_acc))
        + ", test_loss: "
        + str("{:.4f}".format(test_loss))
    )
    logging.info(f"max_train_acc: {max_acc_t:.3%}")
    logging.info(f"max_test_acc: {max_acc_v:.3%}")
    logging.info(
        f"max_lr: "
        + str("{:.6f}".format(max(lr)))
        + ", min_lr: "
        + str("{:.6f}".format(min(lr)))
    )
    logging.info(f"======================================")
    # 画混淆矩阵
    draw_confusion_matrix(
        label,
        pred,
        ["Absent", "Present"],
        "epoch" + str(epochs + 1) + ",testacc: {:.3%}".format(test_acc),
        pdf_save_path=confusion_matrix_path,
        epoch=epochs + 1,
    )


# ========================/ training and logging info /========================== #
logger_init()
model_name = MyModel.model_name
logging.info("<" + model_name + " + 1 fc layer >")
logging.info(f"# trainset_size =  {trainset_size}")
logging.info(f"# testset_size =  {testset_size}")
logging.info(f"# train_a/p = {train_absent_size}/{train_present_size}")
logging.info(f"# test_a/p ={test_absent_size}/{test_present_size}")
logging.info(f"# batch_size = {batch_size}")
logging.info(f"# train total model = {train_total}")
logging.info(f"# learning_rate = {learning_rate:.6f}")
logging.info(f"# num_epochs = {num_epochs}")
logging.info(f"# padding_size = {padding_size}")
logging.info("# loss_fn = " + loss_type)
logging.info(f"# scheduler = {scheduler}")
logging.info("# optimizer = " + str(optimizer))
logging.info("-------------------------------------")
confusion_matrix_path = r"./confusion_matrix/" + str(
    datetime.now().strftime("%Y-%m%d %H%M")
)
tb_writer = SummaryWriter(
    r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
)

for epoch in range(num_epochs):
    train_model(
        model=MyModel,
        device=DEVICE,
        train_loader=train_loader,
        test_loader=test_loader,
        padding=padding_mask,
        # padding_v=padding_mask_v,
        epochs=epoch,
    )
tb_writer.close()
