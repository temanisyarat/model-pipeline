#!/usr/bin/env python3
"""Test the self-contained model with a .npz file.

Usage:
    python try.py                          # test a random sample
    python try.py data/fredi/aku/aku_orig.npz  # test a specific file
"""

import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from src.model import build_mobile_sign_gru
from src.export import load_weights_from_savedmodel

CLASSES = [
    "aku", "apel", "ayah", "besok", "buku", "dia", "dua",
    "hari ini", "ibu", "kamu", "kuning", "maaf", "merah",
    "nama", "pisang", "salam", "satu", "teman",
    "terima kasih", "tiga",
]

# MediaPipe Pose landmark indices used by the model (from 33 total):
# [0= nose, 11=left_shoulder, 12=right_shoulder,
#  13=left_elbow, 14=right_elbow,
#  15=left_wrist, 16=right_wrist,
#  23=left_hip, 24=right_hip]
# Hand order in the 153-d vector: 9 pose + left hand (21) + right hand (21)
MAX_LEN = 125
OUTPUT_DIR = Path("output")


def load_raw_features(npz_path):
    """Load .npz and extract raw (T, 153) landmarks (NaNs preserved)."""
    data = np.load(npz_path)
    pose = data["pose"]                 # (T, 9, 4)
    hands = data["hands"]               # (T, 2, 21, 3)
    pose_xyz = pose[:, :, :3]           # (T, 9, 3)
    hands_flat = hands.reshape(len(hands), -1)  # (T, 126)
    raw = np.concatenate([
        pose_xyz.reshape(len(pose), -1),
        hands_flat,
    ], axis=1)  # (T, 153)
    return raw.astype(np.float32)


def pad_to(seq, max_len):
    """Pad or truncate to (max_len, D).
    
    Fills unused frames with NaN so the model's internal NaN detection
    sets mask=0 for those positions (matching training behavior).
    """
    T, D = seq.shape
    actual_len = min(T, max_len)
    out = np.empty((max_len, D), dtype=np.float32)
    out.fill(np.nan)
    out[:actual_len] = seq[:actual_len]
    return out, actual_len


def build_model(config):
    """Rebuild model and load weights from SavedModel."""
    model = build_mobile_sign_gru(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        max_len=MAX_LEN,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        l2_reg=config.get("l2_reg", 1e-3),
        conv_filters=config.get("conv_filters", [128, 128]),
        conv_kernel_size=config.get("conv_kernel_size", 5),
        spatial_dropout=config.get("spatial_dropout", 0.2),
        recurrent_dropout=config.get("recurrent_dropout", 0.2),
        use_mask_concat=config.get("use_mask_concat", True),
    )
    # Build
    model(np.random.randn(1, MAX_LEN, config["input_dim"]).astype(np.float32), training=False)
    load_weights_from_savedmodel(model, OUTPUT_DIR / "tf_saved_model")
    return model


def build_wrapper(model, global_mean, global_std, raw_dim=153):
    """Wrap model with raw-input preprocessing."""
    raw_input = tf.keras.Input(shape=(MAX_LEN, raw_dim), dtype="float32", name="raw_input")
    mean_t = tf.constant(global_mean.reshape(1, 1, -1), dtype=tf.float32)
    std_t = tf.constant(global_std.reshape(1, 1, -1), dtype=tf.float32)

    valid_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.equal(x, x), tf.float32), name="nan_mask"
    )(raw_input)
    cleaned = tf.keras.layers.Lambda(
        lambda x: tf.where(tf.not_equal(x, x), 0.0, x), name="nan_to_zero"
    )(raw_input)
    normalized = tf.keras.layers.Lambda(
        lambda x: (x - mean_t) / (std_t + 1e-8), name="normalize"
    )(cleaned)
    preprocessed = tf.keras.layers.Concatenate(axis=-1, name="feat_mask_concat")(
        [normalized, valid_mask]
    )
    return tf.keras.Model(raw_input, model(preprocessed), name="mobilesign_gru_raw")


def main():
    # Pick input file
    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
    else:
        # Pick a random sample
        from src.data import scan_dataset_with_signers
        paths, labels, _ = scan_dataset_with_signers("data")
        idx = np.random.randint(len(paths))
        npz_path = Path(paths[idx])
        label = labels[idx]
        print(f"Random sample: {npz_path.name} (expected: {label})")

    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        sys.exit(1)

    # Load config and stats
    with open(OUTPUT_DIR / "config.json") as f:
        config = json.load(f)

    global_mean = np.load(OUTPUT_DIR / "global_mean.npy")
    global_std = np.load(OUTPUT_DIR / "global_std.npy")
    raw_dim = config["input_dim"] // 2

    print(f"Input: {npz_path}")
    print(f"Config: max_len={config['max_len']}, raw_dim={raw_dim}, classes={config['num_classes']}")

    # Build model
    print("\nLoading model...")
    model = build_model(config)
    wrapper = build_wrapper(model, global_mean, global_std, raw_dim)

    # Load and prepare data
    raw_feat = load_raw_features(npz_path)
    print(f"Raw features: {raw_feat.shape}, NaNs: {np.isnan(raw_feat).sum()}")
    padded, actual_len = pad_to(raw_feat, MAX_LEN)
    print(f"Padded to: {padded.shape} ({actual_len} real frames)")

    # Run inference
    logits = wrapper(padded[np.newaxis, ...], training=False).numpy()[0]
    pred_class = int(np.argmax(logits))
    probs = tf.nn.softmax(logits).numpy()

    print(f"\nPrediction: {CLASSES[pred_class]} (class {pred_class})")
    print(f"Confidence: {probs[pred_class]:.4f}")
    print(f"\nTop-5 classes:")
    for i in np.argsort(probs)[::-1][:5]:
        print(f"  {CLASSES[i]:20s}  {probs[i]:.4f}")

    # If we know the expected label
    if len(sys.argv) == 1:
        expected_idx = CLASSES.index(label) if label in CLASSES else -1
        if expected_idx == pred_class:
            print(f"\n✓ Correct! (expected: {label})")
        else:
            print(f"\n✗ Wrong! Expected: {label} (class {expected_idx})")

    print("\nLogits:")
    print(np.array2string(logits, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
