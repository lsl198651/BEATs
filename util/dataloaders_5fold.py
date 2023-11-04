import numpy as np
import random
import os


def get_features(args, train_fold, test_fold):
    npy_path_padded = r"D:\Shilong\murmur\01_dataset\05_5fold\npyFile_padded\npy_files"

    root_path = r"D:\Shilong\murmur\01_dataset\05_5fold"
    npy_path_padded = root_path+r"\npyFile_padded\npy_files"
    index_path = root_path + r"\npyFile_padded\index_files"
    if not os.path.exists(npy_path_padded):
        os.makedirs(npy_path_padded)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    for k in range(5):
        src_fold_root_path = root_path+r"\fold_set_"+str(k)
    # data_Auge(src_fold_root_path)
        for folder in os.listdir(src_fold_root_path):
            dataset_path = os.path.join(src_fold_root_path, folder)
            if folder == "absent":
                pass
            else:
                np.load(npy_path_padded +
                        f"\\{folder}_features_norm_fold{k}.npy", allow_pickle=True)
                np.load(npy_path_padded +
                        f"\\{folder}_labels_norm_fold{k}.npy", allow_pickle=True)
                np.load(npy_path_padded +
                        f"\\{folder}_index_norm_fold{k}.npy", allow_pickle=True)


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
#  =============================/ load index /============================= #

    absent_test_index = np.load(
        npy_path_padded + r"\absent_test_index_norm.npy", allow_pickle=True
    )
    absent_train_index = np.load(
        npy_path_padded + r"\absent_train_index_norm.npy", allow_pickle=True
    )
    present_train_index = np.load(
        npy_path_padded + r"\present_train_index_norm.npy", allow_pickle=True
    )
    present_test_index = np.load(
        npy_path_padded + r"\present_test_index_norm.npy", allow_pickle=True
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
        present_train_features_reserse8 = np.load(
            npy_path_padded + r"\present_train_features_reverse8_norm.npy", allow_pickle=True
        )
        present_train_features_reserse9 = np.load(
            npy_path_padded + r"\present_train_features_reverse9_norm.npy", allow_pickle=True
        )
        present_train_features_reserse11 = np.load(
            npy_path_padded + r"\present_train_features_reverse11_norm.npy", allow_pickle=True
        )
        present_train_features_reserse12 = np.load(
            npy_path_padded + r"\present_train_features_reverse12_norm.npy", allow_pickle=True
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
        present_train_label_reserse8 = np.load(
            npy_path_padded + r"\present_train_label_reverse8_norm.npy", allow_pickle=True
        )
        present_train_label_reserse9 = np.load(
            npy_path_padded + r"\present_train_label_reverse9_norm.npy", allow_pickle=True
        )
        present_train_label_reserse11 = np.load(
            npy_path_padded + r"\present_train_label_reverse11_norm.npy", allow_pickle=True
        )
        present_train_label_reserse12 = np.load(
            npy_path_padded + r"\present_train_label_reverse12_norm.npy", allow_pickle=True
        )
        # ------------------------/ load index /-------------------------- #
        present_train_index_8 = np.load(
            npy_path_padded + r"\present_train_index_8_norm.npy", allow_pickle=True
        )
        present_train_index_9 = np.load(
            npy_path_padded + r"\present_train_index_9_norm.npy", allow_pickle=True
        )
        present_train_index_11 = np.load(
            npy_path_padded + r"\present_train_index_11_norm.npy", allow_pickle=True
        )
        present_train_index_12 = np.load(
            npy_path_padded + r"\present_train_index_12_norm.npy", allow_pickle=True
        )
        # -
        present_train_index_reverse = np.load(
            npy_path_padded + r"\present_train_index_reverse_norm.npy", allow_pickle=True
        )
        present_train_index_reverse8 = np.load(
            npy_path_padded + r"\present_train_index_reverse8_norm.npy", allow_pickle=True
        )
        present_train_index_reverse9 = np.load(
            npy_path_padded + r"\present_train_index_reverse9_norm.npy", allow_pickle=True
        )
        present_train_index_reverse11 = np.load(
            npy_path_padded + r"\present_train_index_reverse11_norm.npy", allow_pickle=True
        )
        present_train_index_reverse12 = np.load(
            npy_path_padded + r"\present_train_index_reverse12_norm.npy", allow_pickle=True
        )

        if args.trainset_balence is True:
            absent_size = int(
                (
                    present_train_features.shape[0]
                    + present_train_features_8.shape[0]
                    + present_train_features_9.shape[0]
                    + present_train_features_11.shape[0]
                    + present_train_features_12.shape[0]
                    + present_train_features_reserse.shape[0],
                    +present_train_features_reserse8.shape[0],
                    +present_train_features_reserse9.shape[0],
                    +present_train_features_reserse11.shape[0],
                    +present_train_features_reserse12.shape[0]
                )
                * args.ap_ratio
            )
            List_train = random.sample(
                range(1, absent_train_features.shape[0]), absent_size
            )
            absent_train_features = absent_train_features[List_train]
            absent_train_label = absent_train_label[List_train]
            absent_train_index = absent_train_index[List_train]
        train_label = np.hstack(
            (
                absent_train_label,
                present_train_label,
                present_train_label_8,
                present_train_label_9,
                present_train_label_11,
                present_train_label_12,
                present_train_label_reserse,
                present_train_label_reserse8,
                present_train_label_reserse9,
                present_train_label_reserse11,
                present_train_label_reserse12,
            )
        )
        train_features = np.vstack(
            (
                absent_train_features,
                present_train_features,
                present_train_features_8,
                present_train_features_9,
                present_train_features_11,
                present_train_features_12,
                present_train_features_reserse,
                present_train_features_reserse8,
                present_train_features_reserse9,
                present_train_features_reserse11,
                present_train_features_reserse12,
            )
        )
        train_index = np.hstack(
            (
                absent_train_index,
                present_train_index,
                present_train_index_8,
                present_train_index_9,
                present_train_index_11,
                present_train_index_12,
                present_train_index_reverse,
                present_train_index_reverse8,
                present_train_index_reverse9,
                present_train_index_reverse11,
                present_train_index_reverse12,
            )
        )
    else:
        if args.trainset_balence is True:
            absent_size = int(present_train_features.shape[0] * args.ap_ratio)
            List_train = random.sample(
                range(1, absent_train_features.shape[0]), absent_size
            )
            absent_train_features = absent_train_features[List_train]
            absent_train_label = absent_train_label[List_train]
            absent_train_index = absent_train_index[List_train]
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
        train_index = np.hstack(
            (
                absent_train_index,
                present_train_index
            )
        )
    test_label = np.hstack((absent_test_label, present_test_label))
    test_features = np.vstack((absent_test_features, present_test_features))
    test_index = np.hstack((absent_test_index, present_test_index))

    return (
        train_features,
        train_label,
        test_features,
        test_label,
        train_index,
        test_index,
    )
