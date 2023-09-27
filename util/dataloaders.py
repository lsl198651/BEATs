import numpy as np
import random


def get_features(args):
    npy_path_padded = r"D:\Shilong\murmur\01_dataset\01_s1s2\npyFile_padded\normalized"

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
        present_train_features_8 = np.load(
            npy_path_padded + r"\present_train_features_8_norm.npy", allow_pickle=True
        )
        present_train_features_12 = np.load(
            npy_path_padded + r"\present_train_features_11_norm.npy", allow_pickle=True
        )
        present_train_label_8 = np.load(
            npy_path_padded + r"\present_train_label_8_norm.npy", allow_pickle=True
        )
        present_train_label_12 = np.load(
            npy_path_padded + r"\present_train_label_11_norm.npy", allow_pickle=True
        )

        # absent_size = int(
        #     (
        #         present_train_features.shape[0]
        #         + present_train_features_12.shape[0]
        #         + present_train_features_8.shape[0]
        #     )
        #     * args.ap_ratio
        # )
        # List_train = random.sample(
        #     range(1, absent_train_features.shape[0]), absent_size
        # )
        # absent_train_features = absent_train_features[List_train]
        # absent_train_label = absent_train_label[List_train]
        train_label = np.hstack(
            (
                absent_train_label,
                present_train_label,
                present_train_label_8,
                present_train_label_12,
            )
        )

        train_features = np.vstack(
            (
                absent_train_features,
                present_train_features,
                present_train_features_8,
                present_train_features_12,
            )
        )
    else:
        # absent_size = int(present_train_features.shape[0] * args.ap_ratio)
        # List_train = random.sample(
        #     range(1, absent_train_features.shape[0]), absent_size
        # )
        # absent_train_features = absent_train_features[List_train]
        # absent_train_label = absent_train_label[List_train]
        train_label = np.hstack(
            (
                absent_train_label,
                present_train_label,
            )
        )

        train_features = np.vstack(
            (
                absent_train_features,
                present_train_features,
            )
        )

    if args.testset_balance is True:
        absent_size = int(present_test_features.shape[0] * args.ap_ratio)
        List_test = random.sample(
            range(1, absent_test_features.shape[0]), absent_size)
        absent_test_features = absent_test_features[List_test]
        absent_test_label = absent_test_label[List_test]
    else:
        pass

    test_label = np.hstack((absent_test_label, present_test_label))
    test_features = np.vstack(
        (
            absent_test_features,
            present_test_features,
        )
    )
    return (
        train_features.astype(float),
        train_label.astype(int),
        test_features.astype(float),
        test_label.astype(int),
    )
