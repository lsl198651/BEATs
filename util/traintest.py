from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from util.BEATs_def import draw_confusion_matrix
from torch import optim
from transformers import optimization
import logging
from sklearn.metrics import confusion_matrix


def train_test(
    model,
    train_loader,
    test_loader,
    padding,
    optimizer,
    args=None,
    tb_writer=None,
    matrix_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 放到设备中
    # train model
    lr = []
    train_acc_list = []
    test_acc_list = []

    # for amp
    scaler = GradScaler()
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
    if args.loss_type == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()  # BCELoss+sigmoid
    elif args.loss_type == "CE":
        loss_fn = nn.CrossEntropyLoss()  # 内部会自动加上Softmax层

    for epoch in range(args.num_epochs):

        train_loss = 0
        correct_t = 0

        model.train()
        optimizer.zero_grad()
        for data_t, label_t in train_loader:
            data_t, label_t = data_t.to(device), label_t.to(device)
            padding = padding.to(device)
            # with autocast(device_type='cuda', dtype=torch.float16):# 这函数害人呀，慎用
            predict = model(data_t, padding)
            # if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            #     pred = torch.max(predict, dim=1)[0]
            #     loss = loss_fn(predict[:, 1], label_t.float())
            # else:
            loss_t = loss_fn(predict, label_t.long())

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss_t.backward()
            optimizer.step()
            train_loss += loss_t.item()
            pred_t = predict.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            pred_t = torch.squeeze(pred_t)
            target_t = label_t.type(torch.int64)
            correct_t += pred_t.eq(target_t).sum().item()

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
                # optimizer.zero_grad()
                predict_v = model(data_v, padding)
                # recall = recall_score(y_hat, y)
                # if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                #     # pred_v = torch.max(predict_v, dim=1)[0]
                #     loss = loss_fn(predict_v[:, 1], label_t.float())
                # else:
                loss_v = loss_fn(predict_v, label_t.long())
                test_loss += loss_v.item()
                pred_v = predict_v.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                pred_v = torch.squeeze(pred_v)
                target_v = label_v.type(torch.int64)
                correct_v += pred_v.eq(target_v).sum().item()
                pred.extend(pred_v.cpu().tolist())
                label.extend(label_v.cpu().tolist())
                # from sklearn.metrics import classification_report
                # y_pred = [0, 1, 0, 0, 1]
                # y_correct = [1, 1, 0, 1, 1]
                # print(classification_report(y_correct, y_pred))

        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)

        # 更新权值
        test_loss /= len(test_loader.dataset.label)
        train_loss /= len(train_loader.dataset.label)
        train_acc = correct_t / len(train_loader.dataset.label)
        test_acc = correct_v / len(test_loader.dataset.label)
        # acc = accuracy_score(pred , target)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        max_train_acc = max(train_acc_list)
        max_test_acc = max(test_acc_list)

        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("test_acc", test_acc, epoch)
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("test_loss", test_loss, epoch)
        tb_writer.add_scalar("learning_rate", lr_now, epoch)

        # a=save_info(num_epochs, epoch, loss, test_acc, test_loss)
        logging.info(f"epoch:  {epoch + 1}/{args.num_epochs}")
        logging.info(f"learning_rate: {lr_now:.6f}")
        logging.info(
            f"train_acc: {train_acc:.3%} train_loss: {train_loss:.4f}")
        logging.info(f"test_acc: {test_acc:.3%} test_loss:{test_loss:.4f}")
        logging.info(f"max_train_acc:{max_train_acc:.3%}")
        logging.info(f"max_test_acc: {max_test_acc:.3%}")
        logging.info(f"max_lr: {max(lr):.6f} , min_lr: {min(lr):.6f}")
        logging.info("======================================")
        # 画混淆矩阵
        draw_confusion_matrix(
            label,
            pred,
            ["Absent", "Present"],
            "epoch" + str(epoch + 1) + ",testacc: {:.3%}".format(test_acc),
            pdf_save_path=matrix_path,
            epoch=epoch + 1,
        )
