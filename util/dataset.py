import os
from BEATs_def import get_wav_data
import numpy as np
# ---------------------------
# -----/ 此文件暂时不用 /----
# ---------------------------


def build_dataset(is_train, args):
    root = os.path.join(args.data_path, "train" if is_train else "val")


def save_data(args):
    absent_train_features, absent_train_label = get_wav_data(
        args.absent_train_path, args
    )  # absent
    absent_test_features, absent_test_label = get_wav_data(
        args.absent_test_path, args
    )  # absent
    present_train_features, present_train_label = get_wav_data(
        args.present_train_path, args
    )  # present
    present_test_features, present_test_label = get_wav_data(
        args.present_test_path, args
    )  # present
    present_test_features_8, present_test_label_8 = get_wav_data(
        args.present_test_path_8, args
    )  # present
    present_test_features_12, present_test_label_12 = get_wav_data(
        args.present_test_path_12, args
    )  # present
    # # # ========================/ save as npy file /========================== #
    np.save(args.npy_path_padded +
            r"\absent_train_features.npy", absent_train_features)
    np.save(args.npy_path_padded +
            r"\absent_test_features.npy", absent_test_features)
    np.save(
        args.npy_path_padded + r"\present_train_features.npy", present_train_features
    )
    np.save(args.npy_path_padded +
            r"\present_test_features.npy", present_test_features)
    np.save(
        args.npy_path_padded + r"\present_test_features_8.npy", present_test_features_8
    )
    np.save(
        args.npy_path_padded + r"\present_test_features_12.npy",
        present_test_features_12,
    )

    np.save(args.npy_path_padded + r"\absent_train_label.npy", absent_train_label)
    np.save(args.npy_path_padded + r"\absent_test_label.npy", absent_test_label)
    np.save(args.npy_path_padded +
            r"\present_train_label.npy", present_train_label)
    np.save(args.npy_path_padded + r"\present_test_label.npy", present_test_label)
    np.save(args.npy_path_padded +
            r"\present_test_label_8.npy", present_test_label_8)
    np.save(args.npy_path_padded +
            r"\present_test_label_12.npy", present_test_label_12)
