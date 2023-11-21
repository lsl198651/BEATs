import argparse
import torch
import torch.profiler
import logging
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from BEATs import BEATs_Pre_Train_itere3
from util.dataloaders import get_features
from util.dataloaders_5fold import fold5_dataloader
from util.traintest import train_test
from util.BEATs_def import (MyDataset, logger_init, DatasetClass)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=512,
                    help="args.batch_size for training")
parser.add_argument("--learning_rate", type=float,
                    default=0.0000001, help="learning_rate for training")
parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
parser.add_argument("--layers", type=int, default=3, help="layers number")
parser.add_argument("--loss_type", type=str, default="FocalLoss",
                    help="loss function", choices=["BCE", "CE", "FocalLoss"])
parser.add_argument("--scheduler_flag", type=str, default=None,
                    help="the dataset used", choices=["cos", "cos_warmup"],)
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
parser.add_argument("--model", type=str,
                    default="BEATs_iter3_plus_AS2M", help="the model used")
parser.add_argument("--ap_ratio", type=float, default=1.0,
                    help="ratio of absent and present")
parser.add_argument("--confusion_matrix_path", type=float,
                    default=1.0, help="ratio of absent and present",)
parser.add_argument("--beta", type=float, default=(0.9, 0.98), help="beta")
parser.add_argument("--cross_evalue", type=bool, default=False)
parser.add_argument("--train_fold", type=list, default=['0', '1', '2', '4'])
parser.add_argument("--test_fold", type=list, default=['4'])
parser.add_argument("--setType", type=str, default=r"\06_new5fold")
args = parser.parse_args()

train_features, train_label, test_features, test_label, train_index, test_index = fold5_dataloader(
    args.train_fold, args.test_fold, args.Data_Augmentation, args.setType)
# ========================/ setup loader /========================== #
if args.samplerWeight == True:
    weights = [3 if label == 1 else 1 for label in train_label]
    Data_sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
    train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index),
                              sampler=Data_sampler, batch_size=args.args.batch_size, drop_last=True,)
else:
    train_loader = DataLoader(DatasetClass(wavlabel=train_label, wavdata=train_features, wavidx=train_index),
                              batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,)

val_loader = DataLoader(
    DatasetClass(wavlabel=test_label,
                 wavdata=test_features, wavidx=test_index),
    batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
)

# ========================/ dataset size /========================== #
train_present_size = np.sum(train_label == 1)
train_absent_size = np.sum(train_label == 0)
test_present_size = np.sum(test_label == 1)
test_absent_size = np.sum(test_label == 0)
trainset_size = train_label.shape[0]
testset_size = test_label.shape[0]

# ========================/ setup padding /========================== #
padding_size = train_features.shape[1]  # 3500
padding = torch.zeros(
    args.batch_size, padding_size
).bool()  # we randomly mask 75% of the input patches,
padding_mask = torch.Tensor(padding)

MyModel = BEATs_Pre_Train_itere3(args=args)

# ========================/ setup optimizer /========================== #
if args.train_total == False:       # tmd 谁给我这么写的！！！！！！
    for param in MyModel.BEATs.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MyModel.parameters()),
                                  lr=args.learning_rate, betas=args.beta,)
else:
    optimizer = torch.optim.AdamW(MyModel.parameters(),
                                  lr=args.learning_rate, betas=args.beta,)

# ========================/ setup scaler /========================== #
logger_init()
logging.info(f"{args.model} + {args.layers} fc layer")
logging.info(f"# Batch_size = {args.batch_size}")
logging.info(f"# Num_epochs = {args.num_epochs}")
logging.info(f"# Learning_rate = {args.learning_rate:.1e}")
logging.info(f"# lr_scheduler = {args.scheduler_flag}")
logging.info(f"# Padding_size = {padding_size}")
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
    padding=padding_mask,
    optimizer=optimizer,
    args=args,
)
