import numpy as np
import random
import os
# args,


def get_features(train_fold: list, test_fold: list):
    npy_path_padded = r"D:\Shilong\murmur\01_dataset\05_5fold\npyFile_padded\npy_files"

    root_path = r"D:\Shilong\murmur\01_dataset\05_5fold"
    npy_path_padded = root_path+r"\npyFile_padded\npy_files"
    train_feature_dic = {}
    train_labels_dic = {}
    train_index_dic = {}
    for k in train_fold:
        src_fold_root_path = root_path+r"\fold_set_"+k
        train_feature_dic[k] = {}
        train_labels_dic[k] = {}
        train_index_dic[k] = {}
    # data_Auge(src_fold_root_path)
        for folder in os.listdir(src_fold_root_path):
            # if folder == "absent":
            #     train_absent_features = np.load(npy_path_padded +
            #                             f"\\{folder}_features_norm_fold{k}.npy", allow_pickle=True)
            #     train_absnet_label = np.load(npy_path_padded +
            #                         f"\\{folder}_labels_norm_fold{k}.npy", allow_pickle=True)
            #     train_absnet_index = np.load(npy_path_padded +
            #                         f"\\{folder}_index_norm_fold{k}.npy", allow_pickle=True)
            # else:

            train_feature_dic[k][folder] = np.load(npy_path_padded +
                                                   f"\\{folder}_features_norm_fold{k}.npy", allow_pickle=True)
            train_labels_dic[k][folder] = np.load(npy_path_padded +
                                                  f"\\{folder}_labels_norm_fold{k}.npy", allow_pickle=True)
            train_index_dic[k][folder] = np.load(npy_path_padded +
                                                 f"\\{folder}_index_norm_fold{k}.npy", allow_pickle=True)
    test_feature_dic = {}
    test_labels_dic = {}
    test_index_dic = {}
    for k in test_fold:
        test_feature_dic[k] = {}
        test_labels_dic[k] = {}
        test_index_dic[k] = {}
        src_fold_root_path = root_path+r"\fold_set_"+k
        for folder in os.listdir(src_fold_root_path):
            # if folder == "absent":
            #     test_absnet_features = np.load(npy_path_padded +
            #                             f"\\{folder}_features_norm_fold{k}.npy", allow_pickle=True)
            #     test_absnet_label = np.load(npy_path_padded +
            #                         f"\\{folder}_labels_norm_fold{k}.npy", allow_pickle=True)
            #     test_absnet_index = np.load(npy_path_padded +
            #                         f"\\{folder}_index_norm_fold{k}.npy", allow_pickle=True)
            # else:
            test_feature_dic[k][folder] = np.load(npy_path_padded +
                                                  f"\\{folder}_features_norm_fold{k}.npy", allow_pickle=True)
            test_labels_dic[k][folder] = np.load(npy_path_padded +
                                                 f"\\{folder}_labels_norm_fold{k}.npy", allow_pickle=True)
            test_index_dic[k][folder] = np.load(npy_path_padded +
                                                f"\\{folder}_index_norm_fold{k}.npy", allow_pickle=True)
    return train_feature_dic, train_labels_dic, train_index_dic, test_feature_dic, test_labels_dic, test_index_dic, os.listdir(src_fold_root_path)


if __name__ == "__main__":
    train_folder = ['0', '1', '2', '3']
    test_folder = ['4']
    train_feature_dic, train_labels_dic, train_index_dic, test_feature_dic, test_labels_dic, test_index_dic, data_class = get_features(
        train_folder, test_folder)
    for k in train_folder:
        for n in data_class:
            print(train_feature_dic[k][n].shape)
            # print(train_labels_dic[k][n].shape)
            # print(train_index_dic[k][n].shape)

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
        )
    )
    test_features = np.vstack(
        (
            test_feature_dic[test_folder[0]][data_class[0]],
            test_feature_dic[test_folder[0]][data_class[1]],
            test_feature_dic[test_folder[0]][data_class[2]],
            test_feature_dic[test_folder[0]][data_class[3]],
            test_feature_dic[test_folder[0]][data_class[4]],
            test_feature_dic[test_folder[0]][data_class[5]],
            test_feature_dic[test_folder[0]][data_class[6]],
            test_feature_dic[test_folder[0]][data_class[7]],
            test_feature_dic[test_folder[0]][data_class[8]],
            test_feature_dic[test_folder[0]][data_class[9]],

        )
    )
    test_label = np.hstack(
        (
            test_labels_dic[test_folder[0]][data_class[0]],
            test_labels_dic[test_folder[0]][data_class[1]],
            test_labels_dic[test_folder[0]][data_class[2]],
            test_labels_dic[test_folder[0]][data_class[3]],
            test_labels_dic[test_folder[0]][data_class[4]],
            test_labels_dic[test_folder[0]][data_class[5]],
            test_labels_dic[test_folder[0]][data_class[6]],
            test_labels_dic[test_folder[0]][data_class[7]],
            test_labels_dic[test_folder[0]][data_class[8]],
            test_labels_dic[test_folder[0]][data_class[9]],

        )
    )
    test_index = np.hstack(
        (
            test_index_dic[test_folder[0]][data_class[0]],
            test_index_dic[test_folder[0]][data_class[1]],
            test_index_dic[test_folder[0]][data_class[2]],
            test_index_dic[test_folder[0]][data_class[3]],
            test_index_dic[test_folder[0]][data_class[4]],
            test_index_dic[test_folder[0]][data_class[5]],
            test_index_dic[test_folder[0]][data_class[6]],
            test_index_dic[test_folder[0]][data_class[7]],
            test_index_dic[test_folder[0]][data_class[8]],
            test_index_dic[test_folder[0]][data_class[9]],

        )
    )
    print('tataset is ok')
