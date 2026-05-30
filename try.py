#!/usr/bin/env python3
"""Test model_raw_int8.tflite accuracy on data/"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASSES = [
    "aku",
    "apel",
    "ayah",
    "besok",
    "buku",
    "dia",
    "dua",
    "hari ini",
    "ibu",
    "kamu",
    "kuning",
    "maaf",
    "merah",
    "nama",
    "pisang",
    "salam",
    "satu",
    "teman",
    "terima kasih",
    "tiga",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MAX_LEN = 125
RAW_DIM = 153
OUTPUT_DIR = Path("output")


def load_raw_features(npz_path):
    data = np.load(npz_path)
    pose = data["pose"]
    hands = data["hands"]
    pose_xyz = pose[:, :, :3]
    hands_flat = hands.reshape(len(hands), -1)
    raw = np.concatenate(
        [
            pose_xyz.reshape(len(pose), -1),
            hands_flat,
        ],
        axis=1,
    )
    return raw.astype(np.float32)


def pad_to(seq, max_len):
    T, D = seq.shape
    actual_len = min(T, max_len)
    out = np.empty((max_len, D), dtype=np.float32)
    out.fill(np.nan)
    out[:actual_len] = seq[:actual_len]
    return out


def get_expected_label(npz_path):
    return npz_path.parent.name


def scan_data(data_dir):
    root = Path(data_dir)
    samples = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        for class_dir in sorted(person_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in CLASS_TO_IDX:
                continue
            for npz_file in sorted(class_dir.glob("*.npz")):
                samples.append((npz_file, class_name))
    return samples


def main():
    tflite_path = OUTPUT_DIR / "model_raw.tflite"
    if not tflite_path.exists():
        print(f"Model not found: {tflite_path}")
        sys.exit(1)

    print(f"Loading TFLite model from {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(
        f"Input  shape: {input_details[0]['shape']}  dtype: {input_details[0]['dtype']}"
    )
    print(
        f"Output shape: {output_details[0]['shape']} dtype: {
            output_details[0]['dtype']
        }"
    )

    samples = scan_data("data")
    print(f"Total test samples: {len(samples)}")
    if not samples:
        print("No samples found. Check data/ directory.")
        sys.exit(1)

    correct = 0
    total = 0
    class_correct = {c: 0 for c in CLASSES}
    class_total = {c: 0 for c in CLASSES}
    errors = []

    for npz_path, expected_label in samples:
        raw_feat = load_raw_features(npz_path)
        padded = pad_to(raw_feat, MAX_LEN)
        input_data = padded[np.newaxis, ...].astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]["index"])[0]
        pred_idx = int(np.argmax(logits))

        expected_idx = CLASS_TO_IDX[expected_label]
        is_correct = pred_idx == expected_idx
        correct += is_correct
        class_correct[expected_label] += is_correct
        class_total[expected_label] += 1
        total += 1

        if not is_correct:
            errors.append((npz_path, expected_label, CLASSES[pred_idx]))

    for npz_path, exp, pred in errors:
        print(f"  FAIL {npz_path}: expected={exp}, predicted={pred}")

    print(f"\n{'=' * 50}")
    print(f"  Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
    print(f"{'=' * 50}")
    print(f"\nPer-class accuracy:")
    for c in CLASSES:
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            print(f"  {c:20s}  {class_correct[c]:3d}/{class_total[c]:3d}  {acc:5.2f}%")
        else:
            print(f"  {c:20s}  no samples")


if __name__ == "__main__":
    main()
