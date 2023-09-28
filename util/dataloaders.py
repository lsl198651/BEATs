import numpy as np
import random


def get_features(args):
    npy_path_padded = r"D:\Shilong\murmur\01_dataset\01_s1s2\npyFile_padded\normalized"
# ------------------------/ load features /-------------------------- #
    absent_train_features = np.load(
        npy_path_padded + r"\absent_train_features_norm.npy", allow_pickle=True
    )
    absent_test_features = np.load(
        npy_path_padded + r"\absent_test_features_norm.npy", allow_pickle=True
    )
    present_train_features = np.load(
        npy_path_padded + r"\present_train_features_norm.npy", allow_pickle=True
    )
    present_test_features = np.load(
        npy_path_padded + r"\present_test_features_norm.npy", allow_pickle=True
    )
# ------------------------/ load labels /-------------------------- #
    absent_train_label = np.load(
        npy_path_padded + r"\absent_train_label_norm.npy", allow_pickle=True
    )
    absent_test_label = np.load(
        npy_path_padded + r"\absent_test_label_norm.npy", allow_pickle=True
    )
    present_train_label = np.load(
        npy_path_padded + r"\present_train_label_norm.npy", allow_pickle=True
    )
    present_test_label = np.load(
        npy_path_padded + r"\present_test_label_norm.npy", allow_pickle=True
    )

    if args.Data_Augmentation is True:
        # ------------------------/ load features /-------------------------- #
        present_train_features_8 = np.load(
            npy_path_padded + r"\present_train_features_8_norm.npy", allow_pickle=True
        )
        present_train_features_9 = np.load(
            npy_path_padded + r"\present_train_features_9_norm.npy", allow_pickle=True
        )
        present_train_features_11 = np.load(
            npy_path_padded + r"\present_train_features_11_norm.npy", allow_pickle=True
        )
        present_train_features_12 = np.load(
            npy_path_padded + r"\present_train_features_12_norm.npy", allow_pickle=True
        )
        present_train_features_reserse = np.load(
            npy_path_padded + r"\present_train_features_reverse_norm.npy", allow_pickle=True
        )
        # ------------------------/ load labels /-------------------------- #
        present_train_label_8 = np.load(
            npy_path_padded + r"\present_train_label_8_norm.npy", allow_pickle=True
        )
        present_train_label_9 = np.load(
            npy_path_padded + r"\present_train_label_9_norm.npy", allow_pickle=True
        )
        present_train_label_11 = np.load(
            npy_path_padded + r"\present_train_label_11_norm.npy", allow_pickle=True
        )
        present_train_label_12 = np.load(
            npy_path_padded + r"\present_train_label_12_norm.npy", allow_pickle=True
        )
        present_train_label_reserse = np.load(
            npy_path_padded + r"\present_train_label_reverse_norm.npy", allow_pickle=True
        )

        if args.trainset_balence is True:
            absent_size = int(
                (
                    present_train_features.shape[0]
                    + present_train_features_8.shape[0]
                    + present_train_features_9.shape[0]
                    + present_train_features_11.shape[0]
                    + present_train_features_12.shape[0]
                )
                * args.ap_ratio
            )
            List_train = random.sample(
                range(1, absent_train_features.shape[0]), absent_size
            )
            absent_train_features = absent_train_features[List_train]
            absent_train_label = absent_train_label[List_train]
        train_label = absent_train_label + present_train_label + present_train_label_8 + \
            present_train_label_9 + present_train_label_11+present_train_label_12
        train_features = absent_train_features + present_train_features + present_train_features_8 + \
            present_train_features_9 + present_train_features_11 + present_train_features_12

    else:
        if args.trainset_balence is True:
            absent_size = int(present_train_features.shape[0] * args.ap_ratio)
            List_train = random.sample(
                range(1, absent_train_features.shape[0]), absent_size
            )
            absent_train_features = absent_train_features[List_train]
            absent_train_label = absent_train_label[List_train]
        train_label = absent_train_label + present_train_label

        train_features = absent_train_features + present_train_features

    test_label = absent_test_label+present_test_label
    test_features = absent_test_features + present_test_features

#  =============================/ load index /============================= #
    npz_path = r"D:\Shilong\murmur\01_dataset\01_s1s2\npyFile_padded\normalized\npz_files"

    absent_test_index = np.load(
        npz_path + r"\absent_test_index_norm.npy", allow_pickle=True
    )
    absent_train_index = np.load(
        npz_path + r"\absent_train_index_norm.npy", allow_pickle=True
    )
    present_train_index = np.load(
        npz_path + r"\present_train_index_norm.npy", allow_pickle=True
    )
    present_test_index = np.load(
        npz_path + r"\present_test_index_norm.npy", allow_pickle=True
    )
    if args.Data_Augmentation is True:
        present_train_index_8 = np.load(
            npz_path + r"\present_train_index_8_norm.npy", allow_pickle=True
        )
        present_train_index_9 = np.load(
            npz_path + r"\present_train_index_9_norm.npy", allow_pickle=True
        )
        present_train_index_11 = np.load(
            npz_path + r"\present_train_index_11_norm.npy", allow_pickle=True
        )
        present_train_index_12 = np.load(
            npz_path + r"\present_train_index_12_norm.npy", allow_pickle=True
        )
        present_train_index_reverse = np.load(
            npz_path + r"\present_train_index_reverse_norm.npy", allow_pickle=True
        )

        train_index = absent_train_index + present_train_index + present_train_index_8 + \
            present_train_index_9 + present_train_index_11 + present_train_index_12
    else:
        train_index = absent_train_index + present_train_index

    test_index = absent_test_index + present_test_index

    return (
        train_features.astype(float),
        train_label.astype(int),
        test_features.astype(float),
        test_label.astype(int),
        train_index,
        test_index,
    )
