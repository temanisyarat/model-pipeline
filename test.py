import json
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def extract_raw_features(npz_path):
    data = np.load(npz_path)
    pose = data["pose"]
    hands = data["hands"]
    T = pose.shape[0]
    pose_xyz = pose[:, :, :3]
    hands_flat = hands.reshape(T, -1)
    feat = np.concatenate([pose_xyz.reshape(T, -1), hands_flat], axis=1)
    return feat.astype(np.float32)


def get_label(npz_path, label_classes):
    class_name = npz_path.parent.name
    return label_classes.index(class_name) if class_name in label_classes else -1


def main():
    output_dir = Path("output")
    config_path = output_dir / "config.json"
    tflite_path = output_dir / "model_raw.tflite"

    with open(config_path) as f:
        config = json.load(f)

    label_classes = config["label_classes"]
    max_len = config["max_len"]
    raw_dim = config["input_dim"] // 2

    import tensorflow as tf

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    npz_paths = sorted(Path("data").rglob("*.npz"))
    print(f"Found {len(npz_paths)} NPZ files")

    random.Random(42).shuffle(npz_paths)

    all_preds = []
    all_labels = []
    all_paths = []

    for npz_path in npz_paths:
        label_idx = get_label(npz_path, label_classes)
        if label_idx < 0:
            continue

        raw_feat = extract_raw_features(npz_path)
        T = min(raw_feat.shape[0], max_len)
        input_arr = np.full((1, max_len, raw_dim), np.nan, dtype=np.float32)
        input_arr[0, :T] = raw_feat[:T]

        interpreter.set_tensor(input_details[0]["index"], input_arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred_idx = int(np.argmax(output[0]))

        all_preds.append(pred_idx)
        all_labels.append(label_idx)
        all_paths.append(str(npz_path))

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_classes, digits=4))

    print(f"\nConfusion Matrix:")
    print(np.array2string(cm, precision=0, suppress_small=True))

    results = {
        "accuracy": float(acc),
        "total_samples": len(all_labels),
        "label_classes": label_classes,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    signer_accs = {}
    for p, l, pred in zip(all_paths, all_labels, all_preds):
        signer = Path(p).parts[1]
        signer_accs.setdefault(signer, {"correct": 0, "total": 0})
        signer_accs[signer]["total"] += 1
        if pred == l:
            signer_accs[signer]["correct"] += 1
    print(f"\nPer-signer accuracy:")
    for s in sorted(signer_accs):
        c = signer_accs[s]
        sa = c["correct"] / c["total"]
        print(f"  {s:<12} {sa:.4f} ({c['correct']}/{c['total']})")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_classes,
        yticklabels=label_classes,
    )
    plt.title(f"Confusion Matrix — model_raw.tflite (Accuracy: {acc:.2%})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrix.png")
    print(f"\nConfusion matrix saved to {output_dir / 'test_confusion_matrix.png'}")
    print(f"Results saved to {output_dir / 'test_results.json'}")

    mispredicted = [
        (p, l, path)
        for p, l, path in zip(all_preds, all_labels, all_paths)
        if p != l
    ]
    if mispredicted:
        print(f"\nMispredictions ({len(mispredicted)} total):")
        for p, l, path in mispredicted[:20]:
            print(f"  true={label_classes[l]:<15} pred={label_classes[p]:<15}  {path}")
        if len(mispredicted) > 20:
            print(f"  ... and {len(mispredicted) - 20} more")


if __name__ == "__main__":
    main()
