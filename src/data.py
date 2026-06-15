import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from src.config import MAX_LEN, MODEL_INPUT_DIM, BATCH_SIZE


def extract_features_with_mask(npz_path):
    data = np.load(npz_path)
    pose = data["pose"]
    hands = data["hands"]
    T = pose.shape[0]
    pose_xyz = pose[:, :, :3]
    hands_flat = hands.reshape(T, -1)
    feat = np.concatenate(
        [
            pose_xyz.reshape(T, -1),
            hands_flat,
        ],
        axis=1,
    )
    mask = (~np.isnan(feat)).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0).astype(np.float32)
    return feat, mask


def pad_or_truncate(seq, max_len):
    T, D = seq.shape
    actual_len = min(T, max_len)
    out = np.zeros((max_len, D), dtype=np.float32)
    out[:actual_len] = seq[:actual_len]
    return out, actual_len


def scan_dataset_with_signers(root_dir):
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
            for npz in sorted(class_dir.glob("*.npz")):
                paths.append(str(npz))
                labels.append(class_name)
                signer_ids.append(signer_id)
    return paths, labels, signer_ids


def split_by_signers(paths, labels, signer_ids, val_signers):
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
    return (
        train_paths,
        train_labels,
        train_signers,
        val_paths,
        val_labels,
        val_signers_list,
    )


def compute_global_stats(train_paths, max_len):
    all_feats = []
    all_masks = []
    for path in train_paths:
        feat, mask = extract_features_with_mask(path)
        T = min(len(feat), max_len)
        all_feats.append(feat[:T])
        all_masks.append(mask[:T])
    all_feats = np.concatenate(all_feats, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    masked_sum = np.sum(all_feats * all_masks, axis=0, keepdims=True)
    masked_count = np.sum(all_masks, axis=0, keepdims=True) + 1e-8
    mean = (masked_sum / masked_count).astype(np.float32)
    diff = (all_feats - mean) * all_masks
    squared_diff = diff**2
    masked_var = np.sum(squared_diff, axis=0, keepdims=True) / masked_count
    std = (np.sqrt(masked_var) + 1e-8).astype(np.float32)
    return mean, std


def parse_npz(path, label_idx, global_mean, global_std):
    def py_parse(path_bytes, label_np, mean_np, std_np):
        path_str = path_bytes.numpy().decode("utf-8")
        label = label_np.numpy().astype(np.int64)
        mean_vals = mean_np.numpy()
        std_vals = std_np.numpy()
        feat, mask = extract_features_with_mask(path_str)
        feat = (feat - mean_vals) / (std_vals + 1e-8)
        feat_concat = np.concatenate([feat, mask], axis=1)
        feat_padded, _ = pad_or_truncate(feat_concat, MAX_LEN)
        return feat_padded.astype(np.float32), label.astype(np.int64)

    feat_shape = tf.TensorShape([MAX_LEN, MODEL_INPUT_DIM])
    label_shape = tf.TensorShape(())
    feat, label = tf.py_function(
        py_parse, [path, label_idx, global_mean, global_std], [tf.float32, tf.int64]
    )
    feat.set_shape(feat_shape)
    label.set_shape(label_shape)
    return feat, label


def augment_sequence(feat, label):
    feat_shape = tf.shape(feat)
    # batch_size, T, D = feat_shape[0], feat_shape[1], feat_shape[2]
    T = feat_shape[1]
    noise = tf.random.normal(tf.shape(feat), mean=0.0, stddev=0.03)
    feat = feat + noise
    channel_drop_prob = tf.random.uniform(tf.shape(feat))
    channel_drop = tf.cast(channel_drop_prob > 0.10, tf.float32)
    feat = feat * channel_drop
    mask_len = tf.random.uniform([], 5, 20, dtype=tf.int32)
    max_start = tf.maximum(1, T - mask_len)
    mask_start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
    time_mask = tf.ones([T], dtype=tf.float32)
    time_mask = tf.tensor_scatter_nd_update(
        time_mask,
        tf.reshape(tf.range(mask_start, mask_start + mask_len), [-1, 1]),
        tf.zeros([mask_len]),
    )
    time_mask = tf.reshape(time_mask, [1, T, 1])
    feat = feat * time_mask
    return feat, label


def create_tf_dataset(
    paths,
    labels,
    le,
    global_mean,
    global_std,
    batch_size=32,
    shuffle=False,
    repeat=False,
    augment=False,
):
    label_indices = le.transform(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))
    if shuffle:
        ds = ds.shuffle(len(paths))
    if repeat:
        ds = ds.repeat()
    mean_ds = tf.data.Dataset.from_tensor_slices([global_mean]).repeat(len(paths))
    std_ds = tf.data.Dataset.from_tensor_slices([global_std]).repeat(len(paths))
    ds = tf.data.Dataset.zip((ds, mean_ds, std_ds))
    ds = ds.map(
        lambda d, m, s: parse_npz(d[0], d[1], m, s), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def build_tf_dataloaders(
    root_dir,
    max_len=150,
    batch_size=32,
    k_folds=9,
    current_fold=0,
    le_fitted=None,
    augment=True,
    paths=None,
    labels=None,
    signer_ids=None,
):
    if paths is None or labels is None or signer_ids is None:
        paths, labels, signer_ids = scan_dataset_with_signers(root_dir)

    le = le_fitted if le_fitted is not None else LabelEncoder().fit(labels)
    unique_signers = sorted(set(signer_ids))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = list(kf.split(unique_signers))
    val_indices = folds[current_fold][1]
    val_signers = [unique_signers[i] for i in val_indices]
    (
        train_paths,
        train_labels,
        train_signers,
        val_paths,
        val_labels,
        val_signers_list,
    ) = split_by_signers(paths, labels, signer_ids, val_signers)
    global_mean, global_std = compute_global_stats(train_paths, max_len)
    train_ds = create_tf_dataset(
        train_paths,
        train_labels,
        le,
        global_mean,
        global_std,
        batch_size=BATCH_SIZE,
        shuffle=True,
        repeat=True,
        augment=augment,
    )
    val_ds = create_tf_dataset(
        val_paths,
        val_labels,
        le,
        global_mean,
        global_std,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
    )
    signer_info = {
        "global_mean": global_mean,
        "global_std": global_std,
        "val_signers": val_signers,
    }
    return train_ds, val_ds, le, len(le.classes_), signer_info
