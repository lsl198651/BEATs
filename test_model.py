# 使用测试集测试模型
import os
import torch
from model.senet.se_resnet import se_resnet6
import utils
from torch.utils.data import DataLoader
from util.BEATs_def import DatasetClass_t
from util.BEATs_def import FocalLoss, get_wav_data
from util.helper_code import get_murmur


def run_model(model_folder, data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # ---------------load model----------------
    print('Loading Challenge model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = se_resnet6().to(device)
    check_path = os.path.join(model_folder, 'model_best_.pth.tar')
    model = utils.load_checkpoint(check_path, model)
    all_output = []
    all_target = []
    # 对每个patient进行预测
    for patient_folder in os.listdir(data_folder):
        target = get_murmur(patient_folder)
        all_target.append(target)
        # 对patient的每个文件夹进行预测
        for dir in os.listdir(patient_folder):
            # 读取测试集标签和特征
            features_t, label_t, names_t, index_t, data_id_t, feat_t = get_wav_data(
                dir, num=0)
            test_loader = DataLoader(DatasetClass_t(wavlabel=label_t, wavdata=features_t, wavidx=index_t),
                                     batch_size=1, shuffle=False, pin_memory=True, num_workers=3)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0001, betas=(0.9, 0.98))
            loss_fn = FocalLoss()
            model.eval()
            label_list = []
            with torch.no_grad():
                for batch_idx, (data_t, target_t, index_t, feat_t) in enumerate(test_loader):
                    data_t, target_t, index_t, feat_t = data_t.to(device), target_t.to(
                        device), index_t.to(device), feat_t.to(device)
                    optimizer.zero_grad()
                    predict_t = model(data_t, feat_t)
                    loss_t = loss_fn(predict_t, target_t.long())
                    test_loss += loss_t.item()
                    pred_t = predict_t.max(1, keepdim=True)[1]
                    label = 1 if pred_t >= .5 else 0
                label_list.append(label)
        output = 1 if label_list.sum() > 0 else 0
        all_output.append(output)
    print('Done.')


if __name__ == '__main__':
    data_folder = r'D:\Shilong\murmur\01_dataset\all_data\training_data'
    model_folder = r'D:\Shilong\murmur\00_Code\LM\beats1\SE_ResNet6\MyModels'
    output_folder = r'D:\Shilong\murmur\00_Code\LM\beats1\SE_ResNet6\output'
    run_model(model_folder, data_folder, output_folder)
