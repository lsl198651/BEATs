from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from util.BEATs_def import draw_confusion_matrix
from torch import optim
from transformers import optimization
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import confusion_matrix


def train_test(
    model,
    train_loader,
    test_loader,
    padding,
    epochs,
    optimizer,
    args=None,
):
    
    tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
    confusion_matrix_path = r"./confusion_matrix/" + str(datetime.now().strftime("%Y-%m%d %H%M"))

    lr=[]
    max_test_acc=[]
    max_train_acc=[]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 放到设备中

    
    # for amp
# ============lr scheduler================
    # scaler = GradScaler()
    warm_up_ratio = 0.1
    total_steps = len(train_loader) * args.num_epochs

    if args.scheduler_flag == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
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
    model.train()
# ============ training ================
    for epoch_num in range(args.num_epochs):
        # train model
        model.train()
        train_loss = 0
        correct_t = 0
        train_len_=0

        for data_t, label_t in train_loader:
            data_t, label_t ,padding= data_t.to(device), label_t.to(device), padding.to(device)

            # with autocast(device_type='cuda', dtype=torch.float16):# 这函数害人呀，慎用
            predict_t = model(data_t, padding)

            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                pred = torch.max(predict_t, dim=1)[0]
                loss = loss_fn(predict_t[:, 1], label_t.float())
            else:
                loss = loss_fn(predict_t, label_t.long())

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred_t = predict_t.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            train_len+=len(label_t)
            correct_t += pred_t.eq(label_t.view_as(pred_t)).sum().item()

        if args.scheduler_flag is not None:
            scheduler.step()

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
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    # pred_v = torch.max(predict_v, dim=1)[0]
                    loss = loss_fn(predict_v[:, 1], label_t.float())
                else:
                    loss_v = loss_fn(predict_v, label_v.long())
                pred_v = predict_v.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                test_loss+=loss_v.item() 
                correct_v += pred_v.eq(label_v.view_as(pred_v)).sum().item()
                pred.extend(pred_v.cpu().tolist())
                label.extend(label_v.cpu().tolist())

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
        max_test_acc_value = max_test_acc[max_train_acc.index(max_train_acc_value)]

        tb_writer.add_scalar("train_acc", train_acc , epochs)
        tb_writer.add_scalar("test_acc", test_acc , epochs)
        tb_writer.add_scalar("train_loss", train_loss, epochs)
        tb_writer.add_scalar("test_loss", test_loss, epochs)
        tb_writer.add_scalar("learning_rate", lr_now, epochs)

        # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
        logging.info(f"epoch: {epochs + 1}/{args.num_epochs}")
        logging.info(f"learning_rate: {lr_now:.6f}")
        logging.info(f"train_acc: {train_acc:.3%} train_loss: {train_loss:.4f}")
        logging.info(f"max_train_acc: {max_train_acc_value:.3%}")
        logging.info(f"max_test_acc: {max_test_acc_value:.3%}")
        logging.info(f"max_lr:{max(lr):.6f}, min_lr:{min(lr):.6f}")
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
