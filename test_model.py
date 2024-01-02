# 使用测试集测试模型
import os
import numpy as np
import torch
import torch.nn as nn
from model.senet.se_resnet import se_resnet6
import utils
from torch.utils.data import DataLoader
from util.BEATs_def import DatasetClass
from util.BEATs_def import FocalLoss

def run_model(model_folder, data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # ---------------load model----------------
    print('Loading Challenge model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = se_resnet6().to(device)
    check_path = os.path.join(model_folder, 'model_best_{}.pth.tar'.format(i))
    model = utils.load_checkpoint(check_path, model)
    murmur_classes = ['Present',  'Absent']
    # 读取测试集标签和特征
    test_label,test_features,test_index,test_index,test_ebd = test_loader(data_folder)
    test_loader = DataLoader(DatasetClass(wavlabel=test_label,wavdata=test_features, wavidx=test_index, wavebd=test_ebd),
                batch_size=1, shuffle=False,pin_memory=True, num_workers=3)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001, betas=(0.9,0.98))
    loss_fn= FocalLoss()
    model.eval()
    with torch.no_grad():
        for batch_idx,data_v, label_v, index_v, feat_v, embeding_v in enumerate(test_loader):
            data_v, label_v, index_v, feat_v =data_v.to(device), label_v.to(device), index_v.to(device), feat_v.to(device) 
            optimizer.zero_grad()
            predict_v = model(data_v, feat_v)
            loss_v = loss_fn(predict_v, label_v.long())
            # get the index of the max log-probability
            pred_v = predict_v.max(1, keepdim=True)[1]
            test_loss += loss_v.item()
            pred_v = pred_v.squeeze(1)

    print('Done.')

if __name__ == '__main__':
    data_folder=r'D:\Shilong\murmur\01_dataset\all_data\training_data'
    model_folder=r'D:\Shilong\murmur\00_code\HearHeart\model'
    output_folder = r'D:\Shilong\murmur\00_code\HearHeart\output'
    run_model(model_folder, data_folder, output_folder)