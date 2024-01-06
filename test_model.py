# 使用测试集测试模型
import os
import torch
from model.senet.se_resnet import se_resnet6
import utils
from torch.utils.data import DataLoader
from util.BEATs_def import DatasetClass_t
from util.BEATs_def import FocalLoss, get_wav_data
from util.helper_code import get_murmur, find_patient_files, load_patient_data
from torcheval.metrics.functional import binary_auprc, binary_auroc, binary_f1_score, binary_confusion_matrix, binary_accuracy, binary_precision, binary_recall
import numpy as np


def run_model(model_folder, data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # ---------------load model----------------
    print('Loading Challenge model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = se_resnet6().to(device)
    check_path = os.path.join(
        model_folder, 'se_resnet6v2_model_best_2.pth.tar')
    model = utils.load_checkpoint(check_path, model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98))
    loss_fn = FocalLoss()
    all_output = []
    all_target = []
    # 对每个patient进行预测
    for patient_folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, patient_folder)
        patient_files = find_patient_files(folder_path)
        current_patient_data = load_patient_data(patient_files)
        target = get_murmur(current_patient_data)
        if target == "Unknown":
            continue
        elif target == "Present":
            all_target.append(1)
        else:
            all_target.append(0)

        # 对patient的每个文件夹进行预测
        for root, dir, file in os.walk(folder_path):  # 一个patient的文件夹
            for subdir in dir:  # 每个location的文件夹
                dir_path = os.path.join(root, subdir)
            # 读取测试集标签和特征
                features_t, label_t, names_t, index_t, data_id_t, feat_t = get_wav_data(
                    dir_path, num=0)
                test_loader = DataLoader(DatasetClass_t(wavlabel=np.array(label_t), wavdata=np.array(
                    features_t), wavidx=np.array(index_t)), batch_size=len(label_t), shuffle=False, pin_memory=True, num_workers=3)
                model.eval()
                label_list = []
                test_loss = 0
                with torch.no_grad():
                    for batch_idx, (data_t, target_t, index_t, feat_t) in enumerate(test_loader):
                        data_t, target_t, index_t, feat_t = data_t.to(device), target_t.to(
                            device), index_t.to(device), feat_t.to(device)
                        optimizer.zero_grad()
                        predict_t = model(data_t, feat_t)
                        loss_t = loss_fn(predict_t, target_t.long())
                        test_loss += loss_t.item()
                        pred_t = predict_t.max(1, keepdim=True)[1]
                        label = 1 if np.mean(
                            pred_t.cpu().tolist()) >= .5 else 0
                label_list.append(label)
            output = 1 if np.sum(label_list) > 0 else 0
        all_output.append(output)
        # 性能测量
    test_patient_input, test_patient_target = torch.as_tensor(
        all_output), torch.as_tensor(all_target)
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
    test_PPV = binary_precision(test_patient_input, test_patient_target)
    test_TPR = binary_recall(test_patient_input, test_patient_target)
    print('test_patient_acc: ', test_patient_acc)
    print('test_TPR: ', test_TPR)
    print('test_PPV: ', test_PPV)
    print('test_patient_f1: ', test_patient_f1)
    print('test_patient_auprc: ', test_patient_auprc)
    print('test_patient_auroc: ', test_patient_auroc)
    print('test_patient_cm: ', test_patient_cm)

    print('Done.')


if __name__ == '__main__':
    data_folder = r'D:\Shilong\murmur\01_dataset\validset_4k\Absent'
    model_folder = r'D:\Shilong\murmur\00_Code\LM\beats1\SE_ResNet6\MyModels'
    output_folder = r'D:\Shilong\murmur\00_Code\LM\beats1\SE_ResNet6\output'
    run_model(model_folder, data_folder, output_folder)
