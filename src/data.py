import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import MAX_LEN, INPUT_DIM, BATCH_SIZE


def extract_features(npz_path):
    """Load .npz and combine landmarks into flat feature vector."""
    data = np.load(npz_path)
    pose = data['pose']
    face = data['face']
    hands = data['hands']

    T = pose.shape[0]
    pose_xyz = pose[:, :, :3]
    hands_flat = hands.reshape(T, -1)

    feat = np.concatenate([
        pose_xyz.reshape(T, -1),
        face.reshape(T, -1),
        hands_flat,
    ], axis=1)

    return feat.astype(np.float32)


def pad_or_truncate(seq, max_len):
    T, D = seq.shape
    actual_len = min(T, MAX_LEN)
    out = np.zeros((MAX_LEN, D), dtype=np.float32)
    out[:actual_len] = seq[:actual_len]
    return out, actual_len


def impute_nan(seq, col_means=None):
    """Fill NaN with linear interpolation."""
    seq = seq.copy()
    for col in range(seq.shape[1]):
        col_data = seq[:, col]
        nan_mask = np.isnan(col_data)
        if nan_mask.all():
            fallback = col_means[col] if col_means is not None else 0.0
            seq[:, col] = fallback
            continue
        if nan_mask.any():
            x = np.arange(len(col_data))
            seq[:, col] = np.interp(x, x[~nan_mask], col_data[~nan_mask])
    return seq


def scan_dataset_with_signers(root_dir):
    """Scan folder with signer subdirectories."""
    root = Path(root_dir)
    paths, labels, signer_ids = [], [], []

    for signer_dir in sorted(root.iterdir()):
        if not signer_dir.is_dir():
            continue
        signer_id = signer_dir.name

        for class_dir in sorted(signer_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            npz_files = sorted(class_dir.glob('*.npz'))

            for npz in npz_files:
                paths.append(str(npz))
                labels.append(class_name)
                signer_ids.append(signer_id)

    return paths, labels, signer_ids


def scan_dataset(root_dir):
    """Scan folder without signer subdirectories."""
    root = Path(root_dir)
    paths, labels = [], []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for npz in sorted(class_dir.glob('*.npz')):
            paths.append(str(npz))
            labels.append(class_dir.name)

    return paths, labels


def split_by_signers(paths, labels, signer_ids, val_signers):
    """Split data by signers (leave-one-signer-out)."""
    train_paths, train_labels, train_signers = [], [], []
    val_paths, val_labels, val_signers_list = [], [], []

    for path, label, sid in zip(paths, labels, signer_ids):
        if sid in val_signers:
            val_paths.append(path)
            val_labels.append(label)
            val_signers_list.append(sid)
        else:
            train_paths.append(path)
            train_labels.append(label)
            train_signers.append(sid)

    return train_paths, train_labels, train_signers, val_paths, val_labels, val_signers_list


def compute_global_stats(train_paths, max_len):
    """Compute mean & std from all valid frames in the training set."""
    all_frames = []
    for path in train_paths:
        feat = extract_features(path)
        feat = impute_nan(feat, col_means=None)
        T = min(len(feat), max_len)
        all_frames.append(feat[:T])
    all_frames = np.concatenate(all_frames, axis=0)
    mean = all_frames.mean(axis=0, keepdims=True).astype(np.float32)
    std = (all_frames.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return mean, std


def parse_npz(path, label_idx, global_mean, global_std):
    def py_parse(path_bytes, label_np, mean_np, std_np):
        path_str = path_bytes.numpy().decode('utf-8')
        label = label_np.numpy().astype(np.int64)
        mean_vals = mean_np.numpy()
        std_vals = std_np.numpy()

        feat = extract_features(path_str)
        feat = impute_nan(feat, col_means=mean_vals.flatten())
        feat = (feat - mean_vals) / (std_vals + 1e-8)
        feat_padded, _ = pad_or_truncate(feat, MAX_LEN)

        return feat_padded.astype(np.float32), label.astype(np.int64)

    feat_shape = tf.TensorShape([MAX_LEN, INPUT_DIM])
    label_shape = tf.TensorShape(())

    feat, label = tf.py_function(py_parse, [path, label_idx, global_mean, global_std],
                                [tf.float32, tf.int64])
    feat.set_shape(feat_shape)
    label.set_shape(label_shape)
    return feat, label


def create_tf_dataset(paths, labels, le, global_mean, global_std, batch_size=32,
                      shuffle=False, repeat=False):
    label_indices = le.transform(labels)

    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))
    if shuffle:
        ds = ds.shuffle(len(paths))
    if repeat:
        ds = ds.repeat()

    mean_ds = tf.data.Dataset.from_tensor_slices([global_mean]).repeat(len(paths))
    std_ds = tf.data.Dataset.from_tensor_slices([global_std]).repeat(len(paths))
    ds = tf.data.Dataset.zip((ds, mean_ds, std_ds))

    ds = ds.map(lambda d, m, s: parse_npz(d[0], d[1], m, s),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_tf_dataloaders(root_dir, max_len=150, batch_size=32, k_folds=5, current_fold=0,
                        le_fitted=None):
    paths, labels, signer_ids = scan_dataset_with_signers(root_dir)

    le = le_fitted if le_fitted is not None else LabelEncoder().fit(labels)
    import random
    from sklearn.model_selection import KFold

    unique_signers = sorted(set(signer_ids))

    kf = KFold(n_splits=k_folds, shuffle=False)
    folds = list(kf.split(unique_signers))
    val_indices = folds[current_fold][1]
    val_signers = [unique_signers[i] for i in val_indices]

    train_paths, train_labels, train_signers, val_paths, val_labels, val_signers_list = \
        split_by_signers(paths, labels, signer_ids, val_signers)

    global_mean, global_std = compute_global_stats(train_paths, max_len)

    train_ds = create_tf_dataset(train_paths, train_labels, le, global_mean, global_std,
                                batch_size=BATCH_SIZE, shuffle=True, repeat=True)
    val_ds = create_tf_dataset(val_paths, val_labels, le, global_mean, global_std,
                              batch_size=BATCH_SIZE, shuffle=False)
    signer_info = {
        'global_mean': global_mean,
        'global_std': global_std,
        'val_signers': val_signers
    }

    return train_ds, val_ds, le, len(le.classes_), signer_info