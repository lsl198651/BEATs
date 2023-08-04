import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
import beats_try
import util.BEATs_def as BEATs_def
from BEATs import BEATs_Pre_Train_itere3
import logging
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=128, help="batch_size for training")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning_rate for training")
parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
parser.add_argument("--loss_type", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--DataBalance", type=str, default=None, help="Dataset balance or not")

parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
args = parser.parse_args()


if args.DataBalance == 'bal':
    logging.info('balanced sampler is being used')
    samples_weight = [9 if label == 1 else 1 for data, label in train_set]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(BEATs_def.MyDataset(wavlabel=train_label, wavdata=train_features),
                                            batch_size=args.batch_size, sampler=sampler,drop_last=True,)
else:
    logging.info('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        BEATs_def.MyDataset(wavlabel=train_label, wavdata=train_features),
        batch_size=args.batch_size, drop_last=True,shuffle=True pin_memory=True)        
val_loader = torch.utils.data.DataLoader(
        BEATs_def.MyDataset(wavlabel=test_label, wavdata=test_features),
        batch_size=args.batch_size*2, shuffle=True, drop_last=True, pin_memory=True)

MyModel = BEATs_Pre_Train_itere3()


for epoch in range(args.num_epochs):
    beats_try.train_model(
        model=MyModel,
        train_loader=train_loader,
        test_loader=val_loader,
        padding=padding_mask,
        epochs=epoch,
    )
