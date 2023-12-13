import argparse
import torch
import torch.profiler
import logging
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from model.model_sknet import AudioClassifier
# from BEATs import BEATs_Pre_Train_itere3
# from model.CNN_SENet import AudioClassifier_SENet
# from model.senet.se_resnet import se_resnet18
# from util.dataloaders import get_features
from util.dataloaders_5fold import fold5_dataloader
from util.traintest import train_test
from util.BEATs_def import (logger_init, DatasetClass)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="args.batch_size for training")
    parser.add_argument("--learning_rate", type=float,
                        default=0.0001, help="learning_rate for training")
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="num_epochs")
    parser.add_argument("--layers", type=int, default=3, help="layers number")
    parser.add_argument("--loss_type", type=str, default="FocalLoss",
                        help="loss function", choices=["BCE", "CE", "FocalLoss"])
    parser.add_argument("--scheduler_flag", type=str, default=None,
                        help="the dataset used", choices=["cos", "MultiStepLR"],)
    parser.add_argument("--freqm_value",  type=int, default=0,
                        help="frequency mask max length")
    parser.add_argument("--timem_value", type=int, default=0,
                        help="time mask max length")
    parser.add_argument("--mask", type=bool, default=False,
                        help="number of classes", choices=[True, False])
    parser.add_argument("--trainset_balence", type=bool, default=False,
                        help="balance absent and present in testset", choices=[True, False],)
    parser.add_argument("--Data_Augmentation", type=bool, default=False,
                        help="Add data augmentation", choices=[True, False],)
    parser.add_argument("--train_total", type=bool, default=True,
                        help="use grad_no_requiredn", choices=[True, False],)
    parser.add_argument("--samplerWeight", type=bool, default=False,
                        help="use balanced sampler", choices=[True, False],)
    # TODO改模型名字
    parser.add_argument("--model", type=str, default="skconv+feat",
                        help="the model used")
    parser.add_argument("--ap_ratio", type=float, default=1.0,
                        help="ratio of absent and present")
    parser.add_argument("--beta", type=float, default=(0.9, 0.98), help="beta")
    parser.add_argument("--cross_evalue", type=bool, default=False)
    parser.add_argument("--train_fold", type=list,
                        default=['0', '1', '2', '3'])
    parser.add_argument("--test_fold", type=list, default=['4'])
    parser.add_argument("--setType", type=str, default=r"\11_baseset")
    parser.add_argument("--model_folder", type=str,
                        default=r"D:\Shilong\murmur\00_Code\LM\beats1\BEATs\MyModels")
    args = parser.parse_args()
    # 检测分折重复
    for val in args.test_fold:
        if val in args.train_fold:
            raise ValueError("train_fold and test_fold have same fold")

    train_features, train_label, train_index, train_ebd, test_features,  test_label, test_index, test_ebd = fold5_dataloader(
        args.train_fold, args.test_fold, args.Data_Augmentation, args.setType)
    # ========================/ setup loader /========================== #
    if args.samplerWeight == True:
        weights = [3 if label == 1 else 1 for label in train_label]
        Data_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index, wavebd=train_ebd),
                                  sampler=Data_sampler, batch_size=args.args.batch_size, drop_last=True, num_workers=4)
    else:
        train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index, wavebd=train_ebd),
                                  batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)

    val_loader = DataLoader(
        DatasetClass(wavlabel=test_label,
                     wavdata=test_features, wavidx=test_index, wavebd=test_ebd),
        batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # ========================/ dataset size /========================== #
    train_present_size = np.sum(train_label == 1)
    train_absent_size = np.sum(train_label == 0)
    test_present_size = np.sum(test_label == 1)
    test_absent_size = np.sum(test_label == 0)
    trainset_size = train_label.shape[0]
    testset_size = test_label.shape[0]

    # ========================/ setup padding /========================== #
    # padding_size = train_features.shape[1]  # 3500
    # padding = torch.zeros(
    #     args.batch_size, padding_size
    # ).bool()  # we randomly mask 75% of the input patches,
    # padding_mask = torch.Tensor(padding)

    MyModel = AudioClassifier()
    # MyModel = se_resnet18()

    # ========================/ setup optimizer /========================== #
    if not args.train_total:       # tmd 谁给我这么写的！！！！！！
        for param in MyModel.BEATs.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MyModel.parameters()),
                                      lr=args.learning_rate, betas=args.beta,)
    else:
        optimizer = torch.optim.AdamW(MyModel.parameters(),
                                      lr=args.learning_rate, betas=args.beta,)

    # ========================/ setup scaler /========================== #
    logger_init()
    logging.info(f"{args.model}  ")
    logging.info(f"# Batch_size = {args.batch_size}")
    logging.info(f"# Num_epochs = {args.num_epochs}")
    logging.info(f"# Learning_rate = {args.learning_rate:.1e}")
    logging.info(f"# lr_scheduler = {args.scheduler_flag}")
    # logging.info(f"# Padding_size = {padding_size}")
    logging.info(f"# Loss_fn = {args.loss_type}")
    logging.info(f"# Data Augmentation = {args.Data_Augmentation}")
    logging.info(f"# Trainset_balance = {args.trainset_balence}")
    logging.info(f"# train_total = {args.train_total}")
    logging.info(f"# Masking = {args.mask}")
    logging.info(f"# SetType = {args.setType}")
    logging.info(f"# Train_a/p = {train_absent_size}/{train_present_size}")
    logging.info(f"# Test_a/p = {test_absent_size}/{test_present_size}")
    logging.info(f"# Trainset_size = {trainset_size}")
    logging.info(f"# Testset_size = {testset_size}")
    logging.info(f"# Train_fold = {args.train_fold}")
    logging.info(f"# Test_fold = {args.test_fold}")
    logging.info("# Optimizer = " + str(optimizer))
    logging.info(
        "# ")

    train_test(
        model=MyModel,
        train_loader=train_loader,
        test_loader=val_loader,
        padding=None,
        optimizer=optimizer,
        args=args,
    )
