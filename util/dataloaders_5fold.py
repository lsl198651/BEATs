from cgi import test
import numpy as np
import random
import os
# args,


def get_features(train_fold: list, test_fold: list, set_type: str):
    # npy_path_padded = r"D:\Shilong\murmur\01_dataset\05_5fold\npyFile_padded\npy_files01"
    root_path = r"D:\Shilong\murmur\01_dataset"+set_type
    npy_path_padded = root_path+r"\npyFile_padded\npy_files01"
    train_feature_dic = {}
    train_labels_dic = {}
    train_index_dic = {}
    train_ebd_dic = {}
    for k in train_fold:
        src_fold_root_path = root_path+r"\fold_set_"+k
        train_feature_dic[k] = {}
        train_labels_dic[k] = {}
        train_index_dic[k] = {}
        train_ebd_dic[k] = {}
    # data_Auge(src_fold_root_path)
        train_folders = ['absent', 'present']
        for folder in train_folders:
            train_feature_dic[k][folder] = np.load(npy_path_padded +
                                                   f"\\{folder}_features_norm01_fold{k}.npy", allow_pickle=True)
            train_labels_dic[k][folder] = np.load(npy_path_padded +
                                                  f"\\{folder}_labels_norm01_fold{k}.npy", allow_pickle=True)
            train_index_dic[k][folder] = np.load(npy_path_padded +
                                                 f"\\{folder}_index_norm01_fold{k}.npy", allow_pickle=True)
            train_ebd_dic[k][folder] = np.load(npy_path_padded +
                                               f"\\{folder}_feat_norm01_fold{k}.npy", allow_pickle=True)

    test_feature_dic = {}
    test_labels_dic = {}
    test_index_dic = {}
    test_ebd_dic = {}
    for v in test_fold:
        test_feature_dic[v] = {}
        test_labels_dic[v] = {}
        test_index_dic[v] = {}
        test_ebd_dic[v] = {}
        src_fold_root_path = root_path+r"\fold_set_"+k
        test_folders = ['absent', 'present']
        for folder in test_folders:
            test_feature_dic[v][folder] = np.load(npy_path_padded +
                                                  f"\\{folder}_features_norm01_fold{v}.npy", allow_pickle=True)
            test_labels_dic[v][folder] = np.load(npy_path_padded +
                                                 f"\\{folder}_labels_norm01_fold{v}.npy", allow_pickle=True)
            test_index_dic[v][folder] = np.load(npy_path_padded +
                                                f"\\{folder}_index_norm01_fold{v}.npy", allow_pickle=True)
            test_ebd_dic[v][folder] = np.load(npy_path_padded +
                                              f"\\{folder}_feat_norm01_fold{v}.npy", allow_pickle=True)
    return train_feature_dic, train_labels_dic, train_index_dic, train_ebd_dic, test_feature_dic, test_labels_dic, test_index_dic, test_ebd_dic, train_folders


def fold5_dataloader(train_folder, test_folder, Data_Augmentation, set_type):
    """组合特征并且返回features，label，index"""
    train_feature_dic, train_labels_dic, train_index_dic, train_ebd_dic, test_feature_dic, test_labels_dic, test_index_dic, test_ebd_dic, data_class = get_features(
        train_folder, test_folder, set_type)
    if Data_Augmentation is True:
        train_features = np.vstack(
            (

                train_feature_dic[train_folder[0]][data_class[0]],
                train_feature_dic[train_folder[0]][data_class[1]],
                train_feature_dic[train_folder[0]][data_class[2]],
                train_feature_dic[train_folder[0]][data_class[3]],
                train_feature_dic[train_folder[0]][data_class[4]],
                train_feature_dic[train_folder[0]][data_class[5]],
                train_feature_dic[train_folder[0]][data_class[6]],
                train_feature_dic[train_folder[0]][data_class[7]],
                train_feature_dic[train_folder[0]][data_class[8]],
                train_feature_dic[train_folder[0]][data_class[9]],
                train_feature_dic[train_folder[0]][data_class[10]],
                train_feature_dic[train_folder[1]][data_class[0]],
                train_feature_dic[train_folder[1]][data_class[1]],
                train_feature_dic[train_folder[1]][data_class[2]],
                train_feature_dic[train_folder[1]][data_class[3]],
                train_feature_dic[train_folder[1]][data_class[4]],
                train_feature_dic[train_folder[1]][data_class[5]],
                train_feature_dic[train_folder[1]][data_class[6]],
                train_feature_dic[train_folder[1]][data_class[7]],
                train_feature_dic[train_folder[1]][data_class[8]],
                train_feature_dic[train_folder[1]][data_class[9]],
                train_feature_dic[train_folder[1]][data_class[10]],
                train_feature_dic[train_folder[2]][data_class[0]],
                train_feature_dic[train_folder[2]][data_class[1]],
                train_feature_dic[train_folder[2]][data_class[2]],
                train_feature_dic[train_folder[2]][data_class[3]],
                train_feature_dic[train_folder[2]][data_class[4]],
                train_feature_dic[train_folder[2]][data_class[5]],
                train_feature_dic[train_folder[2]][data_class[6]],
                train_feature_dic[train_folder[2]][data_class[7]],
                train_feature_dic[train_folder[2]][data_class[8]],
                train_feature_dic[train_folder[2]][data_class[9]],
                train_feature_dic[train_folder[2]][data_class[10]],
                train_feature_dic[train_folder[3]][data_class[0]],
                train_feature_dic[train_folder[3]][data_class[1]],
                train_feature_dic[train_folder[3]][data_class[2]],
                train_feature_dic[train_folder[3]][data_class[3]],
                train_feature_dic[train_folder[3]][data_class[4]],
                train_feature_dic[train_folder[3]][data_class[5]],
                train_feature_dic[train_folder[3]][data_class[6]],
                train_feature_dic[train_folder[3]][data_class[7]],
                train_feature_dic[train_folder[3]][data_class[8]],
                train_feature_dic[train_folder[3]][data_class[9]],
                train_feature_dic[train_folder[3]][data_class[10]]
            )
        )
        train_label = np.hstack(
            (
                train_labels_dic[train_folder[0]][data_class[0]],
                train_labels_dic[train_folder[0]][data_class[1]],
                train_labels_dic[train_folder[0]][data_class[2]],
                train_labels_dic[train_folder[0]][data_class[3]],
                train_labels_dic[train_folder[0]][data_class[4]],
                train_labels_dic[train_folder[0]][data_class[5]],
                train_labels_dic[train_folder[0]][data_class[6]],
                train_labels_dic[train_folder[0]][data_class[7]],
                train_labels_dic[train_folder[0]][data_class[8]],
                train_labels_dic[train_folder[0]][data_class[9]],
                train_labels_dic[train_folder[0]][data_class[10]],
                train_labels_dic[train_folder[1]][data_class[0]],
                train_labels_dic[train_folder[1]][data_class[1]],
                train_labels_dic[train_folder[1]][data_class[2]],
                train_labels_dic[train_folder[1]][data_class[3]],
                train_labels_dic[train_folder[1]][data_class[4]],
                train_labels_dic[train_folder[1]][data_class[5]],
                train_labels_dic[train_folder[1]][data_class[6]],
                train_labels_dic[train_folder[1]][data_class[7]],
                train_labels_dic[train_folder[1]][data_class[8]],
                train_labels_dic[train_folder[1]][data_class[9]],
                train_labels_dic[train_folder[1]][data_class[10]],
                train_labels_dic[train_folder[2]][data_class[0]],
                train_labels_dic[train_folder[2]][data_class[1]],
                train_labels_dic[train_folder[2]][data_class[2]],
                train_labels_dic[train_folder[2]][data_class[3]],
                train_labels_dic[train_folder[2]][data_class[4]],
                train_labels_dic[train_folder[2]][data_class[5]],
                train_labels_dic[train_folder[2]][data_class[6]],
                train_labels_dic[train_folder[2]][data_class[7]],
                train_labels_dic[train_folder[2]][data_class[8]],
                train_labels_dic[train_folder[2]][data_class[9]],
                train_labels_dic[train_folder[2]][data_class[10]],
                train_labels_dic[train_folder[3]][data_class[0]],
                train_labels_dic[train_folder[3]][data_class[1]],
                train_labels_dic[train_folder[3]][data_class[2]],
                train_labels_dic[train_folder[3]][data_class[3]],
                train_labels_dic[train_folder[3]][data_class[4]],
                train_labels_dic[train_folder[3]][data_class[5]],
                train_labels_dic[train_folder[3]][data_class[6]],
                train_labels_dic[train_folder[3]][data_class[7]],
                train_labels_dic[train_folder[3]][data_class[8]],
                train_labels_dic[train_folder[3]][data_class[9]],
                train_labels_dic[train_folder[3]][data_class[10]]
            )
        )
        train_index = np.hstack(
            (
                train_index_dic[train_folder[0]][data_class[0]],
                train_index_dic[train_folder[0]][data_class[1]],
                train_index_dic[train_folder[0]][data_class[2]],
                train_index_dic[train_folder[0]][data_class[3]],
                train_index_dic[train_folder[0]][data_class[4]],
                train_index_dic[train_folder[0]][data_class[5]],
                train_index_dic[train_folder[0]][data_class[6]],
                train_index_dic[train_folder[0]][data_class[7]],
                train_index_dic[train_folder[0]][data_class[8]],
                train_index_dic[train_folder[0]][data_class[9]],
                train_index_dic[train_folder[0]][data_class[10]],
                train_index_dic[train_folder[1]][data_class[0]],
                train_index_dic[train_folder[1]][data_class[1]],
                train_index_dic[train_folder[1]][data_class[2]],
                train_index_dic[train_folder[1]][data_class[3]],
                train_index_dic[train_folder[1]][data_class[4]],
                train_index_dic[train_folder[1]][data_class[5]],
                train_index_dic[train_folder[1]][data_class[6]],
                train_index_dic[train_folder[1]][data_class[7]],
                train_index_dic[train_folder[1]][data_class[8]],
                train_index_dic[train_folder[1]][data_class[9]],
                train_index_dic[train_folder[1]][data_class[10]],
                train_index_dic[train_folder[2]][data_class[0]],
                train_index_dic[train_folder[2]][data_class[1]],
                train_index_dic[train_folder[2]][data_class[2]],
                train_index_dic[train_folder[2]][data_class[3]],
                train_index_dic[train_folder[2]][data_class[4]],
                train_index_dic[train_folder[2]][data_class[5]],
                train_index_dic[train_folder[2]][data_class[6]],
                train_index_dic[train_folder[2]][data_class[7]],
                train_index_dic[train_folder[2]][data_class[8]],
                train_index_dic[train_folder[2]][data_class[9]],
                train_index_dic[train_folder[2]][data_class[10]],
                train_index_dic[train_folder[3]][data_class[0]],
                train_index_dic[train_folder[3]][data_class[1]],
                train_index_dic[train_folder[3]][data_class[2]],
                train_index_dic[train_folder[3]][data_class[3]],
                train_index_dic[train_folder[3]][data_class[4]],
                train_index_dic[train_folder[3]][data_class[5]],
                train_index_dic[train_folder[3]][data_class[6]],
                train_index_dic[train_folder[3]][data_class[7]],
                train_index_dic[train_folder[3]][data_class[8]],
                train_index_dic[train_folder[3]][data_class[9]],
                train_index_dic[train_folder[3]][data_class[10]]
            )
        )
    else:
        train_features = np.vstack(
            (
                train_feature_dic[train_folder[0]][data_class[0]],
                train_feature_dic[train_folder[0]][data_class[1]],
                train_feature_dic[train_folder[1]][data_class[0]],
                train_feature_dic[train_folder[1]][data_class[1]],
                train_feature_dic[train_folder[2]][data_class[0]],
                train_feature_dic[train_folder[2]][data_class[1]],
                train_feature_dic[train_folder[3]][data_class[0]],
                train_feature_dic[train_folder[3]][data_class[1]],
            )
        )
        train_label = np.hstack(
            (
                train_labels_dic[train_folder[0]][data_class[0]],
                train_labels_dic[train_folder[0]][data_class[1]],
                train_labels_dic[train_folder[1]][data_class[0]],
                train_labels_dic[train_folder[1]][data_class[1]],
                train_labels_dic[train_folder[2]][data_class[0]],
                train_labels_dic[train_folder[2]][data_class[1]],
                train_labels_dic[train_folder[3]][data_class[0]],
                train_labels_dic[train_folder[3]][data_class[1]]
            )
        )
        train_index = np.hstack(
            (
                train_index_dic[train_folder[0]][data_class[0]],
                train_index_dic[train_folder[0]][data_class[1]],
                train_index_dic[train_folder[1]][data_class[0]],
                train_index_dic[train_folder[1]][data_class[1]],
                train_index_dic[train_folder[2]][data_class[0]],
                train_index_dic[train_folder[2]][data_class[1]],
                train_index_dic[train_folder[3]][data_class[0]],
                train_index_dic[train_folder[3]][data_class[1]]
            )
        )
        train_ebd = np.hstack(
            (
                train_ebd_dic[train_folder[0]][data_class[0]],
                train_ebd_dic[train_folder[0]][data_class[1]],
                train_ebd_dic[train_folder[1]][data_class[0]],
                train_ebd_dic[train_folder[1]][data_class[1]],
                train_ebd_dic[train_folder[2]][data_class[0]],
                train_ebd_dic[train_folder[2]][data_class[1]],
                train_ebd_dic[train_folder[3]][data_class[0]],
                train_ebd_dic[train_folder[3]][data_class[1]]
            )
        )

    test_features = np.vstack(
        (
            test_feature_dic[test_folder[0]]['absent'],
            test_feature_dic[test_folder[0]]['present'],
        )
    )
    test_label = np.hstack(
        (
            test_labels_dic[test_folder[0]]['absent'],
            test_labels_dic[test_folder[0]]['present'],
        )
    )
    test_index = np.hstack(
        (
            test_index_dic[test_folder[0]]['absent'],
            test_index_dic[test_folder[0]]['present'],
        )
    )
    test_ebd = np.hstack(
        (
            test_ebd_dic[test_folder[0]]['absent'],
            test_ebd_dic[test_folder[0]]['present'],
        )
    )
    return train_features, train_label, train_index, train_ebd, test_features,  test_label, test_index, test_ebd
