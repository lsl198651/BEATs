from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from util.BEATs_def import draw_confusion_matrix
from torch import optim
from transformers import optimization
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from util.BEATs_def import FocalLoss, segment_classifier, get_segment_target_list
import logging

import os
import pandas as pd


def train_test(
    model,
    train_loader,
    test_loader,
    padding=None,
    optimizer=None,
    args=None,
):
    error_index_path = r"./error_index/" + \
        str(datetime.now().strftime("%Y-%m%d %H%M"))
    if not os.path.exists(error_index_path):
        os.makedirs(error_index_path)
    tb_writer = SummaryWriter(
        r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
    confusion_matrix_path = r"./confusion_matrix/" + \
        str(datetime.now().strftime("%Y-%m%d %H%M"))

    lr = []
    max_test_acc = []
    max_train_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 放到设备中

    # for amp
# ============lr scheduler================
    # scaler = GradScaler()
    warm_up_ratio = 0.1
    total_steps = len(train_loader) * args.num_epochs

    if args.scheduler_flag == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0)
    elif args.scheduler_flag == "cos_warmup":
        scheduler = optimization.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_ratio * total_steps,
            num_training_steps=total_steps,
        )
# ==========loss function================
    if args.loss_type == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()  # BCELoss+sigmoid
    elif args.loss_type == "CE":
        loss_fn = nn.CrossEntropyLoss()  # 内部会自动加上Softmax层
    elif args.loss_type == "FocalLoss":
        loss_fn = FocalLoss()
    model.train()
# ============ training ================
    for epochs in range(args.num_epochs):
        # train model
        model.train()
        train_loss = 0
        correct_t = 0
        train_len = 0

        for data_t, label_t, index_t in train_loader:
            data_t, label_t, padding, index_t = data_t.to(
                device), label_t.to(device), padding.to(device), index_t.to(device)
            # with autocast(device_type='cuda', dtype=torch.float16):# 这函数害人呀，慎用
            predict_t = model(data_t, padding)
            if args.loss_type == "BCE":
                predict_t2 = torch.argmax(predict_t, dim=1)
                loss = loss_fn(predict_t2.float(), label_t)
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # predict_t2 = predict_t2.squeeze(1)
                correct_t += predict_t2.eq(label_t).sum().item()
                train_len += len(label_t)
            elif args.loss_type == "FocalLoss" or "CE":
                loss = loss_fn(
                    predict_t, label_t.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # get the index of the max log-probability
                pred_t = predict_t.max(1, keepdim=True)[1]
                # label_t = torch.int64(label_t)
                pred_t = pred_t.squeeze(1)
                correct_t += pred_t.eq(label_t).sum().item()
                train_len += len(label_t)
        if args.scheduler_flag is not None:
            scheduler.step()
        # evaluate model
        model.eval()
        label = []
        pred = []
        error_index = []
        result_list_present = []
        test_loss = 0
        correct_v = 0
        with torch.no_grad():
            for data_v, label_v, index_v in test_loader:
                data_v, label_v, padding, index_v = (
                    data_v.to(device),
                    label_v.to(device),
                    padding.to(device),
                    index_v.to(device),
                )
                optimizer.zero_grad()
                predict_v = model(data_v, padding)
                # recall = recall_score(y_hat, y)
                if args.loss_type == "BCE":
                    predict_v2 = torch.argmax(predict_v, dim=1)
                    loss_v = loss_fn(predict_v2.float(), label_t)
                    test_loss += loss_v.item()
                    # predict_v2 = predict_v2.squeeze(1)
                    correct_v += predict_v2.eq(label_v).sum().item()
                    pred.extend(predict_v2.cpu().tolist())
                    label.extend(label_v.cpu().tolist())
                elif args.loss_type == "FocalLoss" or "CE":
                    loss_v = loss_fn(predict_v, label_v.long())
                    # get the index of the max log-probability
                    pred_v = predict_v.max(1, keepdim=True)[1]
                    test_loss += loss_v.item()
                    pred_v = pred_v.squeeze(1)
                    correct_v += pred_v.eq(label_v).sum().item()
                    idx_v = index_v[torch.nonzero(
                        torch.eq(pred_v.ne(label_v), True))]
                    idx_v = idx_v.squeeze()
                    result_list_present.extend(index_v[torch.nonzero(
                        torch.eq(pred_v.eq(1), True))].cpu().tolist())
                    # result_list_present = result_list_present.squeeze()
                    error_index.extend(idx_v.cpu().tolist())
                    pred.extend(pred_v.cpu().tolist())
                    label.extend(label_v.cpu().tolist())
            pd.DataFrame(error_index).to_csv(error_index_path+"/epoch" +
                                             str(epochs+1)+".csv", index=False, header=False)
            location_acc, location_cm, patient_acc, patient_cm = segment_classifier(
                result_list_present)
            segment_cm = confusion_matrix(label, pred, labels=[0, 1])
        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)
        # 更新权值
        test_loss /= len(pred)
        train_loss /= train_len
        train_acc = correct_t / train_len
        test_acc = correct_v / len(pred)
        # acc = accuracy_score(pred , target)
        max_train_acc.append(train_acc)
        max_test_acc.append(test_acc)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(
            max_train_acc_value)]
        tb_writer.add_scalar("train_acc", train_acc, epochs)
        tb_writer.add_scalar("test_acc", test_acc, epochs)
        tb_writer.add_scalar("train_loss", train_loss, epochs)
        tb_writer.add_scalar("test_loss", test_loss, epochs)
        tb_writer.add_scalar("learning_rate", lr_now, epochs)
        # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
        logging.info(f"======================================")
        logging.info(f"epoch: {epochs + 1}/{args.num_epochs}")
        logging.info(f"learning_rate: {lr_now:.1e}")
        logging.info(f"ACC t: {train_acc:.3%} v: {test_acc:.3%}")
        logging.info(f"segment_cm:{segment_cm}")
        logging.info(f"Loss t: {train_loss:.4f} v: {test_loss:.4f}")
        logging.info(
            f"max_acc t: {max_train_acc_value:.3%} v: {max_test_acc_value:.3%}")
        logging.info(f"lr max:{max(lr):.1e} min:{min(lr):.1e}")
        logging.info(f"location_acc:{location_acc:.3%}")
        logging.info(f"location_cm:{location_cm}")
        logging.info(f"patient_acc:{patient_acc:.3%}")
        logging.info(f"patient_cm:{patient_cm}")

        # 画混淆矩阵
        draw_confusion_matrix(
            label,
            pred,
            ["Absent", "Present"],
            "epoch" + str(epochs + 1) + ",testacc: {:.3%}".format(test_acc),
            pdf_save_path=confusion_matrix_path,
            epoch=epochs + 1,
        )
