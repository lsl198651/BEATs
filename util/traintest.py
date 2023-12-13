
import logging
import os
import pandas as pd
import torch
import utils
import torch.nn as nn
from torch import optim, tensor
from datetime import datetime
from transformers import optimization
# import numpy as np
# from sklearn.metrics import confusion_matrix
# from torch.cuda.amp import autocast, GradScaler
# from util.BEATs_def import draw_confusion_matrix, butterworth_low_pass_filter
from torch.utils.tensorboard import SummaryWriter
# from util.BEATs_def import Log_GF
# , get_segment_target_list, FocalLoss_VGG
from util.BEATs_def import FocalLoss, segment_classifier
from torcheval.metrics.functional import binary_auprc, binary_auroc, binary_f1_score, binary_confusion_matrix, binary_accuracy, binary_precision, binary_recall


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
    patient_error_index_path = r"./patient_error_index/" + \
        str(datetime.now().strftime("%Y-%m%d %H%M"))
    if not os.path.exists(error_index_path):
        os.makedirs(error_index_path)
    if not os.path.exists(patient_error_index_path):
        os.makedirs(patient_error_index_path)
    tb_writer = SummaryWriter(
        r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
    confusion_matrix_path = r"./confusion_matrix/" + \
        str(datetime.now().strftime("%Y-%m%d %H%M"))
    lr = []
    max_test_acc = []
    max_train_acc = []
    best_acc = 0.0
    gfcc = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    model = model.to(device)  # 放到设备中
    # for amp
# ============lr scheduler================
    # scaler = GradScaler()
    warm_up_ratio = 0.1
    total_steps = len(train_loader) * args.num_epochs
    if args.scheduler_flag == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0)
    elif args.scheduler_flag == "MultiStep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            [40, 80],
            gamma=0.5
        )
# ==========loss function================
    if args.loss_type == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()  # BCELoss+sigmoid
    elif args.loss_type == "CE":
        normedWeights = [1, 2]
        normedWeights = torch.FloatTensor(normedWeights).to(device)
        loss_fn = nn.CrossEntropyLoss(
            weight=normedWeights)  # 内部会自动加上Softmax层
    elif args.loss_type == "FocalLoss":
        loss_fn = FocalLoss()
    embedding1 = nn.Embedding(5, 10)  # 5个类别，每个类别用10维向量表示
    embedding2 = nn.Embedding(2, 10)  # 2个类别，每个类别用10维向量表示
    embedding3 = nn.Embedding(2, 10)
# ============ training ================
    for epochs in range(args.num_epochs):
        # train model
        model.train()
        train_loss = 0
        correct_t = 0
        train_len = 0
        input_train = []
        target_train = []
        for data_t, label_t, index_t, feat, embeding in train_loader:
            # data_t = butterworth_low_pass_filter(data_t)
            # gfcc = Log_GF(data_t)
            # gfcc = gfcc.to(device)
            # embedings = []
            # for ebed in embeding:
            #     ebd_List = []
            #     ebd_List.append(embedding1(
            #         torch.tensor(ebed//10 % 10)).detach().numpy())
            #     ebd_List.append(embedding2(
            #         torch.tensor(ebed//10 % 10)).detach().numpy())
            #     ebd_List.append(embedding3(
            #         torch.tensor(ebed % 10)).detach().numpy())
            #     embedings.append(ebd_List)
            # embedings = torch.tensor(embedings)
            data_t, label_t,  index_t, feat = data_t.to(
                device), label_t.to(device),  index_t.to(device), feat.to(device)  # , embedings.to(device)
            # with autocast(device_type='cuda', dtype=torch.float16):# 这函数害人呀，慎用
            predict_t = model(data_t, feat)  # , feat
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
                input_train.extend(pred_t.cpu().tolist())
                target_train.extend(label_t.cpu().tolist())
                correct_t += pred_t.eq(label_t).sum().item()
                train_len += len(pred_t)

        # ------------------调库计算指标--------------------------
        train_input, train_target = torch.as_tensor(
            input_train), torch.as_tensor(target_train)
        train_acc = binary_accuracy(train_input, train_target)
        # print(f"train_acc:{train_acc:.2%}")
        # ============ evalue ================
        model.eval()
        label = []
        pred = []
        error_index = []
        result_list_present = []
        test_loss = 0
        correct_v = 0
        with torch.no_grad():
            for data_v, label_v, index_v, feat_v, embeding_v in test_loader:
                # data_v = butterworth_low_pass_filter(data_v)
                # gfcc = Log_GF(data_v)
                # gfcc = gfcc.to(device)
                # embedings_v = []
                # for ebed in embeding_v:
                #     ebd_List = []
                #     ebd_List.append(embedding1(
                #         torch.tensor(ebed//10 % 10)).detach().numpy())
                #     ebd_List.append(embedding2(
                #         torch.tensor(ebed//10 % 10)).detach().numpy())
                #     ebd_List.append(embedding3(
                #         torch.tensor(ebed % 10)).detach().numpy())
                #     embedings_v.append(ebd_List)
                # embedings_v = torch.tensor(embedings_v)
                #
                data_v, label_v, index_v, feat_v = \
                    data_v.to(device), label_v.to(device), index_v.to(
                        device), feat_v.to(device)  # , embedings_v.to(device)
                optimizer.zero_grad()
                # , feat_v, embedings_v
                predict_v = model(data_v, feat_v)
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
                    # idx_v = idx_v.squeeze()
                    result_list_present.extend(index_v[torch.nonzero(
                        torch.eq(pred_v.eq(1), True))].cpu().tolist())
                    # result_list_present = result_list_present.squeeze()
                    try:
                        error_index.extend(idx_v.cpu().tolist())
                    except TypeError:
                        print("TypeError: 'int' object is not iterable")
                    pred.extend(pred_v.cpu().tolist())
                    label.extend(label_v.cpu().tolist())
        if args.scheduler_flag is not None:
            scheduler.step()
        # ------------------调库计算指标--------------------------
        test_input, test_target = torch.as_tensor(pred), torch.as_tensor(label)
        test_auprc = binary_auprc(test_input, test_target)
        test_auroc = binary_auroc(test_input, test_target)
        test_acc = binary_accuracy(test_input, test_target)
        test_f1 = binary_f1_score(test_input, test_target)
        test_cm = binary_confusion_matrix(test_input, test_target)
        # ---------------------------------------------------------
        if best_acc < test_acc:
            utils.save_checkpoint({"epoch": epochs + 1,
                                   "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict()}, args.model, args.test_fold[0], "{}".format(args.model_folder))
        # --------------------------------------------------------
        pd.DataFrame(error_index).to_csv(error_index_path+"/epoch" +
                                         str(epochs+1)+".csv", index=False, header=False)
        location_acc, location_cm, patient_output, patient_target, patient_error_id = segment_classifier(
            result_list_present, args.test_fold, args.setType)  #
        test_patient_input, test_patient_target = torch.as_tensor(
            patient_output), torch.as_tensor(patient_target)
        test_patient_auprc = binary_auprc(
            test_patient_input, test_patient_target)
        test_patient_auroc = binary_auroc(
            test_patient_input, test_patient_target)
        test_patient_acc = binary_accuracy(
            test_patient_input, test_patient_target)
        test_patient_f1 = binary_f1_score(
            test_patient_input, test_patient_target)
        test_patient_cm = binary_confusion_matrix(
            test_patient_input, test_patient_target)
        # 这两个算出来的都是present的
        test_PPV = binary_precision(test_patient_input, test_patient_target)
        test_TPR = binary_recall(test_patient_input, test_patient_target)
        if test_patient_acc > best_acc:
            best_acc = test_patient_acc
        pd.DataFrame(patient_error_id).to_csv(patient_error_index_path+"/epoch" +
                                              str(epochs+1)+".csv", index=False, header=False)
        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)
        # 更新权值
        test_loss /= len(pred)
        train_loss /= train_len
        max_train_acc.append(train_acc)
        max_test_acc.append(test_acc)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(
            max_train_acc_value)]
        # tensorboard
        tb_writer.add_scalar("train_acc", train_acc, epochs)
        tb_writer.add_scalar("test_acc", test_acc, epochs)
        tb_writer.add_scalar("train_loss", train_loss, epochs)
        tb_writer.add_scalar("test_loss", test_loss, epochs)
        tb_writer.add_scalar("learning_rate", lr_now, epochs)
        tb_writer.add_scalar("patient_acc", test_patient_acc, epochs)
        # logging
        logging.info(f"============================")
        logging.info(f"epoch: {epochs + 1}/{args.num_epochs}")
        logging.info(f"learning_rate: {lr_now:.1e}")
        logging.info(f"Loss t: {train_loss:.2e} v: {test_loss:.2e}")
        logging.info(
            f"max_acc t: {max_train_acc_value:.2%} v: {max_test_acc_value:.2%}")
        logging.info(f"lr max:{max(lr):.1e} min:{min(lr):.1e}")
        logging.info(f"train Acc: {train_acc:.2%} \nvalid Acc: {test_acc:.2%}")
        logging.info(f"segment_cm:{test_cm.numpy()}")
        logging.info(f"segments_auroc:{test_auroc:.3f}")
        logging.info(f"segments_auprc:{test_auprc:.3f}")
        logging.info(f"segments_f1_:{test_f1:.3f}")
        # logging.info(f"----------------------------")
        # logging.info(f"location_acc:{location_acc:.2%}")
        # logging.info(f"location_cm:{location_cm}")
        logging.info(f"----------------------------")
        logging.info(f"patient_acc:{test_patient_acc:.2%}")
        logging.info(f"patient_cm:{test_patient_cm.numpy()}")
        logging.info(f"patient_auroc:{test_patient_auroc:.3f}")
        logging.info(f"patient_auprc:{test_patient_auprc:.3f}")
        logging.info(f"patient_f1_:{test_patient_f1:.3f}")
        logging.info(f"patient_PPV:{test_PPV:.3f}")
        logging.info(f"patient_TPR:{test_TPR:.3f}")
        logging.info(f"best_acc:{best_acc:.2%}")
        # 画混淆矩阵
        # draw_confusion_matrix(
        #     test_cm.numpy(),
        #     ["Absent", "Present"],
        #     "epoch" + str(epochs + 1) + ",testacc: {:.3%}".format(test_acc),
        #     pdf_save_path=confusion_matrix_path,
        #     epoch=epochs + 1,
        # )
