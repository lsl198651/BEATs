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

from torch.utils.data.sampler import WeightedRandomSampler
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
from util.BEATs_def import (
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

# ========================/ load npy padded file /========================== #
mask = True
time_stretch = True
Data_Enhancement = False
testset_bal = True
batch_size = 128
learning_rate = 0.001
num_epochs = 100
grad_flag = True
# weight_decay = 0.01
loss_type = "CE"
scheduler = None


npy_path_padded = r"D:\Shilong\murmur\03_circor_states\npyFile_padded"
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

if time_stretch is True:
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


# ========================/ get features & labels /========================== #
# test_features,test_label=get_mel_features(path,absent_patient_id,present_patient_id)
"""
train_features,train_label=get_mel_features(train_path,absent_patient_id,present_patient_id)
test_features,test_label=get_mel_features(test_path,absent_patient_id,present_patient_id)
"""
ap_ratio = 1

if Data_Enhancement is True:
    absent_size = int(
        (
            present_train_features.shape[0]
            + present_train_features_12.shape[0]
            + present_train_features_8.shape[0]
        )
        * ap_ratio
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
    absent_size = int(present_train_features.shape[0] * ap_ratio)
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
if testset_bal is True:
    absent_size = int(present_test_features.shape[0] * ap_ratio)
    List_test = random.sample(range(1, absent_test_features.shape[0]), absent_size)
    absent_test_features = absent_test_features[List_test]
    absent_test_label = absent_test_label[List_test]
else:
    pass

test_label = np.hstack((absent_test_label, present_test_label))
test_features = np.vstack(
    (
        absent_test_features,
        present_test_features,
    )
)

# ========================/ train test /========================== #
train_features = train_features.astype(float)
train_label = train_label.astype(int)
test_features = test_features.astype(float)
test_label = test_label.astype(int)

train_present_size = np.sum(train_label == 1)
train_absent_size = np.sum(train_label == 0)
test_present_size = np.sum(test_label == 1)
test_absent_size = np.sum(test_label == 0)
trainset_size = train_label.shape[0]
testset_size = test_label.shape[0]

# ========================/ HyperParameters /========================== #

padding_size = train_features.shape[1]  # 3500
padding = torch.zeros(
    batch_size, padding_size
).bool()  # we randomly mask 75% of the input patches,
padding_mask = torch.Tensor(padding)

# ========================/ dataloader /========================== #
# 如果label为1，那么对应的该类别被取出来的概率是另外一个类别的2倍
weights = [3 if label == 1 else 1 for label in train_label]
Data_sampler = WeightedRandomSampler(
    weights, num_samples=len(weights), replacement=True
)

train_loader = DataLoader(
    MyDataset(wavlabel=train_label, wavdata=train_features),
    # sampler=Data_sampler,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
)
test_loader = DataLoader(
    MyDataset(wavlabel=test_label, wavdata=test_features),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
print("Dataloader is ok")  # 最后再打印一下新的模型

# ========================/ load model /========================== #
MyModel = BEATs_Pre_Train_itere3()

# ========================/ model Loss fn /========================== #
if loss_type == "BCE":
    loss_fn = nn.BCEWithLogitsLoss()
elif loss_type == "CE":
    loss_fn = nn.CrossEntropyLoss()

# ========================/ setup optimizer /========================== #
if grad_flag is True:
    for param in MyModel.BEATs.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, MyModel.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        # weight_decay=weight_decay,
    )
else:
    optimizer = torch.optim.AdamW(
        MyModel.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        # weight_decay=weight_decay,
    )
# ========================/ setup warmup lr /========================== #
warm_up_ratio = 0.1
total_steps = len(train_loader) * num_epochs

if scheduler is not None:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = optimization.get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warm_up_ratio * total_steps,
    #     num_training_steps=total_steps,
    # )


# ========================/ train model /========================== #
# 定义训练函数
def train_model(
    model,
    train_loader,
    test_loader,
    padding,
    epochs,
    lr=[],
    max_test_acc=[],
    max_train_acc=[],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel.to(device)  # 放到设备中
    # train model
    train_loss = 0
    correct_t = 0
    # for amp
    scaler = GradScaler()
    model.train()
    optimizer.zero_grad()
    for data_t, label_t in train_loader:
        data_t, label_t = data_t.to(device), label_t.to(device)
        padding = padding.to(device)

        # with autocast(device_type='cuda', dtype=torch.float16):# 这函数害人呀，慎用
        predict = model(data_t, padding)
        loss = loss_fn(predict, label_t.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        train_loss += loss.item()
        pred_t = predict.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
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
            # optimizer.zero_grad()
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
    test_loss /= len(test_loader.dataset.label)
    train_loss /= len(train_loader.dataset.label)
    train_acc = correct_t / len(train_loader.dataset.label)
    test_acc = correct_v / len(test_loader.dataset.label)

    max_train_acc.append(train_acc)
    max_test_acc.append(test_acc)
    max_train_acc = max(max_train_acc)
    max_test_acc = max(max_test_acc)

    tb_writer.add_scalar("train_acc", train_acc * 100, epochs)
    tb_writer.add_scalar("test_acc", test_acc * 100, epochs)
    tb_writer.add_scalar("train_loss", train_loss, epochs)
    tb_writer.add_scalar("test_loss", test_loss, epochs)
    tb_writer.add_scalar("learning_rate", lr_now, epochs)

    # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
    logging.info("epoch: " + str(epochs + 1) + "/" + str(num_epochs))
    logging.info("learning_rate: " + str("{:.4f}".format(lr_now)))
    logging.info(
        "train_acc: "
        + str("{:.3%}".format(train_acc))
        + ", train_loss: "
        + str("{:.4f}".format(train_loss))
    )
    logging.info(
        "test_acc: "
        + str("{:.3%}".format(test_acc))
        + ", test_loss: "
        + str("{:.4f}".format(test_loss))
    )
    logging.info(f"max_train_acc: " + str("{:.3%}".format(max_train_acc)))
    logging.info(f"max_test_acc: " + str("{:.3%}".format(max_test_acc)))
    logging.info(
        "max_lr: "
        + str("{:.4f}".format(max(lr)))
        + ", min_lr: "
        + str("{:.4f}".format(min(lr)))
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
logging.info("<<< " + model_name + " - 2 fc layer >>> ")
if mask is True:
    logging.info("Add FrequencyMasking and TimeMasking")
if time_stretch is True:
    logging.info("Add time_stretch 0.8 and time_stretch 1.2")
logging.info("# trainset_size = " + str(trainset_size))
logging.info("# testset_size = " + str(testset_size))
logging.info("# train_a/p = " + "{}/{}".format(train_absent_size, train_present_size))
logging.info("# test_a/p = " + "{}/{}".format(test_absent_size, test_present_size))
logging.info("# batch_size = " + str(batch_size))
logging.info("# learning_rate = " + str(learning_rate))
# logging.info("# weight_decay = " + str(weight_decay))
logging.info("# num_epochs = " + str(num_epochs))
logging.info("# padding_size = " + str(padding_size))
logging.info("# loss_fn = " + loss_type)
logging.info("# scheduler = " + str(scheduler))
logging.info("# optimizer = " + str(optimizer))
logging.info("# comments : ")
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
        train_loader=train_loader,
        test_loader=test_loader,
        padding=padding_mask,
        epochs=epoch,
    )
tb_writer.close()
