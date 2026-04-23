import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from src.config import CONFIG, DATA_DIR, OUTPUT_DIR
from src.data import (
    scan_dataset_with_signers,
    build_tf_dataloaders,
    create_tf_dataset,
)
from src.model import build_mobile_sign_gru
from src.train import train_tf_model
from src.evaluate import (
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    save_evaluation_results,
)
from src.export import (
    convert_saved_model_to_tflite,
    benchmark_tf_model,
    benchmark_tflite_model,
)


def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    K_FOLDS = 5
    CURRENT_FOLD = 0

    all_paths, all_labels, all_signer_ids = scan_dataset_with_signers(DATA_DIR)
    label_encoder_global = LabelEncoder()
    label_encoder_global.fit(all_labels)
    num_classes_global = len(label_encoder_global.classes_)
    print(f"Global classes: {list(label_encoder_global.classes_)}")

    unique_signers = sorted(set(all_signer_ids))
    rng = random.Random(42)
    unique_signers_shuffled = unique_signers.copy()
    rng.shuffle(unique_signers_shuffled)

    TEST_SIGNER = unique_signers_shuffled[-1]
    train_val_signers = unique_signers_shuffled[:-1]

    print(f"Held-out test signer: {TEST_SIGNER}")
    print(f"Train/val signers ({len(train_val_signers)}): {train_val_signers}")

    train_ds, val_ds, label_encoder, num_classes, signer_info = build_tf_dataloaders(
        DATA_DIR,
        max_len=CONFIG['max_len'],
        batch_size=CONFIG['batch_size'],
        k_folds=K_FOLDS,
        current_fold=CURRENT_FOLD,
        le_fitted=label_encoder_global
    )

    test_paths = [p for p, s in zip(all_paths, all_signer_ids) if s == TEST_SIGNER]
    test_labels_list = [l for l, s in zip(all_labels, all_signer_ids) if s == TEST_SIGNER]
    test_ds = create_tf_dataset(
        test_paths, test_labels_list, label_encoder_global,
        signer_info['global_mean'], signer_info['global_std'],
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )
    print(f"Test set size: {len(test_paths)} samples")

    print("\nBuilding TensorFlow MobileSignGRU model...")
    input_dim = CONFIG['input_dim']
    CONFIG['num_classes'] = num_classes

    model = build_mobile_sign_gru(
        input_dim=input_dim,
        num_classes=num_classes,
        max_len=CONFIG['max_len'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        bidirectional=CONFIG['bidirectional'],
    )
    model.summary()

    param_count = model.count_params()
    model_size_mb = sum(w.numpy().nbytes for w in model.weights) / (1024 * 1024)
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    print("\nStarting training pipeline...")

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    if steps_per_epoch < 0:
        n_train = len([s for s in all_signer_ids if s != TEST_SIGNER and s not in signer_info['val_signers']])
        steps_per_epoch = max(1, n_train // CONFIG['batch_size'])

    val_steps = tf.data.experimental.cardinality(val_ds).numpy()
    if val_steps < 0:
        val_steps = None

    model, history_obj = train_tf_model(
        train_ds, val_ds, CONFIG, num_classes, input_dim,
        steps_per_epoch, val_steps
    )

    fold_results = [{'history': history_obj.history, 'accuracy': max(history_obj.history.get('val_accuracy', [0]))}]
    best_fold_idx = 0

    best_history = fold_results[best_fold_idx].get('history', {})
    np.save(OUTPUT_DIR / 'history.npy', best_history)

    CONFIG_for_export = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dim': CONFIG['hidden_dim'],
        'num_layers': CONFIG['num_layers'],
        'dropout': CONFIG['dropout'],
        'bidirectional': CONFIG['bidirectional'],
        'max_len': CONFIG['max_len'],
        'num_params': int(model.count_params()),
        'model_size_mb': round(model_size_mb, 3),
        'label_classes': list(label_encoder.classes_),
    }

    import json
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(CONFIG_for_export, f, indent=2)

    print(f"Model history and CONFIG saved to {OUTPUT_DIR}")

    print("\nTraining complete. Evaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nFinal Test Accuracy (held-out signer {TEST_SIGNER}): {test_acc:.4f}")

    eval_results = evaluate_model(model, test_ds, label_encoder_global)

    print("\nPlotting results...")
    plot_training_history(history_obj, OUTPUT_DIR)
    plot_confusion_matrix(eval_results, OUTPUT_DIR)
    save_evaluation_results(eval_results, OUTPUT_DIR)

    param_count = model.count_params()
    model_size_mb = sum(w.numpy().nbytes for w in model.weights) / (1024 * 1024)
    print(f"\nModel parameters: {param_count:,}")
    print(f"Model size in RAM: {model_size_mb:.2f} MB")

    print("\nBenchmarking TF model inference speed...")
    bench_tf = benchmark_tf_model(model, input_dim, CONFIG['max_len'])
    print(f"  TF Model: {bench_tf['ms_per_sample']:.3f} ms/sample ({bench_tf['fps']:.1f} fps)")

    print("\nExporting to TFLite...")
    saved_model_path = OUTPUT_DIR / 'tf_saved_model'
    model.export(saved_model_path)
    print(f"SavedModel exported to {saved_model_path}")

    tflite_path = OUTPUT_DIR / 'model_int8.tflite'
    tflite_file, tflite_size = convert_saved_model_to_tflite(
        saved_model_path,
        tflite_path,
        CONFIG,
        input_dim,
        quantization='fp16'
    )

    print("Benchmarking TFLite model...")
    tflite_bench = benchmark_tflite_model(tflite_path)
    print(f"TFLite inference: {tflite_bench['mean_ms']:.3f} +/- {tflite_bench['std_ms']:.3f} ms ({tflite_bench['fps']:.1f} fps)")

    print("\n" + "=" * 60)
    print("BISINDO SIGN LANGUAGE RECOGNITION - PIPELINE COMPLETE")
    print("=" * 60)

    print("\nModel Summary:")
    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dim: {CONFIG['hidden_dim']}")
    print(f"   Layers: {CONFIG['num_layers']}")
    print(f"   Bidirectional: {CONFIG['bidirectional']}")
    print(f"   Classes: {num_classes} ({', '.join(label_encoder.classes_)})")

    print("\nModel Sizes:")
    print(f"   TF SavedModel (FP32): {model_size_mb:.2f} MB")
    print(f"   TFLite (FP16):        {tflite_size:.2f} MB")

    print("\nPerformance:")
    print(f"   TF Inference:    {bench_tf['ms_per_sample']:.3f} ms/sample")
    print(f"   TFLite Inference: {tflite_bench['mean_ms']:.3f} ms/sample")

    print("\nTest Evaluation:")
    print(f"   Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"   F1-macro:  {eval_results['f1_macro']:.4f}")
    print(f"   Precision: {eval_results['precision']:.4f}")
    print(f"   Recall:    {eval_results['recall']:.4f}")

    print("\nOutput Files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name}: {size:.2f} MB")


if __name__ == '__main__':
    main()