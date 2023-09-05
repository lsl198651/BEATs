import argparse
import torch
import torch.profiler
import logging
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from datetime import datetime
from torch import optim
from transformers import optimization
from torch.utils.data import DataLoader
from BEATs import BEATs_Pre_Train_itere3
from torch.utils.tensorboard import SummaryWriter
from util.dataloaders import get_features
from util.traintest import train_test
from util.BEATs_def import (MyDataset, logger_init)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=128,
                    help="args.batch_size for training")
parser.add_argument("--learning_rate", type=float,
                    default=0.001, help="learning_rate for training")
parser.add_argument("--num_epochs", type=int, default=500, help="num_epochs")
parser.add_argument("--loss_type", type=str, default="CE",
                    help="loss function", choices=["BCE", "CE"])
parser.add_argument("--scheduler_flag", type=str, default=None,
                    help="the dataset used", choices=["cos", "cos_warmup"],)
parser.add_argument("--freqm_value",  type=int, default=0,
                    help="frequency mask max length")
parser.add_argument("--timem_value", type=int, default=10,
                    help="time mask max length")
parser.add_argument("--mask", type=bool, default=False,
                    help="number of classes", choices=[True, False])
parser.add_argument("--testset_balance", type=bool, default=False,
                    help="balance absent and present in testset", choices=[True, False],)
parser.add_argument("--Data_Augmentation", type=bool, default=False,
                    help="Add data augmentation", choices=[True, False],)
parser.add_argument("--grad_flag", type=bool, default=False,
                    help="use grad_no_requiredn", choices=[True, False],)
parser.add_argument("--samplerWeight", type=bool, default=False,
                    help="use balanced sampler", choices=[True, False],)
parser.add_argument("--layers", type=int, default=1, help="layers number")
parser.add_argument("--train_total_model", type=bool, default=False,
                    help="train total model", choices=[True, False],)
parser.add_argument("--model", type=str,
                    default="BEATs_iter3_plus_AS2M", help="the model used", choices=["BEATs_iter3_plus_AS2M", "BEATs_iter3_plus_AS20K", "BEATs_iter3", "BEATs_iter2"])
parser.add_argument("--ap_ratio", type=float, default=1.0,
                    help="ratio of absent and present")
parser.add_argument("--confusion_matrix_path", type=float,
                    default=1.0, help="ratio of absent and present",)
parser.add_argument("--beta", type=float, default=(0.9, 0.999), help="beta")
args = parser.parse_args()

train_features, train_label, test_features, test_label = get_features(
    args=args)
# ========================/ setup loader /========================== #
if args.samplerWeight == True:
    weights = [3 if label == 1 else 1 for label in train_label]
    Data_sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
    train_loader = DataLoader(MyDataset(wavlabel=train_label, wavdata=train_features),
                              sampler=Data_sampler, batch_size=args.args.batch_size, drop_last=True,)
else:
    train_loader = DataLoader(MyDataset(wavlabel=train_label, wavdata=train_features),
                              batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,)

val_loader = DataLoader(
    MyDataset(wavlabel=test_label, wavdata=test_features),
    batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True,
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
if args.train_total_model == False:
    for param in MyModel.BEATs.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MyModel.parameters()),
                                  lr=args.learning_rate, betas=args.beta,)
else:
    optimizer = torch.optim.AdamW(MyModel.parameters(),
                                  lr=args.learning_rate, betas=args.beta,)

# ========================/ setup scaler /========================== #
logger_init()
logging.info(f"<<< {args.model} - {args.layers} fc layer >>>")
logging.info(f"# Batch_size = {args.batch_size}")
logging.info(f"# Num_epochs = {args.num_epochs}")
logging.info(f"# Learning_rate = {args.learning_rate:.6f}")
logging.info(f"# lr_scheduler = {args.scheduler_flag}")
logging.info(f"# Padding_size = {padding_size}")
logging.info(f"# Loss_fn = {args.loss_type}")
logging.info(f"# Data Augmentation = {args.Data_Augmentation}")
logging.info(f"# Testset_balance = {args.testset_balance}")
logging.info(f"# Masking = {args.mask}")
if args.mask == True:
    logging.info(f"# freqm_value = {args.freqm_value}")
    logging.info(f"# timem_value = {args.timem_value}")
logging.info(f"# wegiht sampler = {args.samplerWeight}")
logging.info(f"# train total model = {args.train_total_model}")
logging.info(f"# Train_a/p = {train_absent_size}/{train_present_size}")
logging.info(f"# Test_a/p = {test_absent_size}/{test_present_size}")
logging.info(f"# Trainset_size = {trainset_size}")
logging.info(f"# Testset_size = {testset_size}")
logging.info("# Optimizer = " + str(optimizer))
logging.info("# Notes : ")
logging.info("-------------------------------------")
confusion_matrix_path = r"./confusion_matrix/" + str(
    datetime.now().strftime("%Y-%m%d %H%M")
)
tb_writer = SummaryWriter(
    r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
)


train_test(
    model=MyModel,
    train_loader=train_loader,
    test_loader=val_loader,
    padding=padding_mask,
    optimizer=optimizer,
    args=args,
    tb_writer=tb_writer,
    matrix_path=confusion_matrix_path,
)
