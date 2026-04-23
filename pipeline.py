import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")


# ## 1. Data Loading & Preprocessing
CONFIG = {
    'max_len': 100,          # Max sequence length (frames)
    'batch_size': 128,       # Batch size
    'input_dim': 252,        # Feature dimension (36 pose + 99 face + 126 hands)
    'num_classes': 5,        # Number of sign classes
    'hidden_dim': 192,       # GRU hidden dimension
    'num_layers': 2,         # Number of GRU layers
    'dropout': 0.25,         # Dropout rate
    'bidirectional': True,   # Use bidirectional GRU
    'learning_rate': 1e-3,   # Initial learning rate
    'epochs': 20,            # Max training epochs
    'patience': 12,          # Early stopping patience
    'label_smoothing': 0.1,  # Label smoothing factor
}

BATCH_SIZE = int(CONFIG['batch_size'])  
MAX_LEN = int(CONFIG['max_len'])       
INPUT_DIM = int(CONFIG['input_dim'])  

CLASSES = ['aku', 'apa', 'dia', 'kamu', 'siapa']

print("Configuration loaded.")
print(f"Classes: {CLASSES}")

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
    """
    Fill NaN with linear interpolation.
    If full-column NaN, use col_means (from training stats)
    instead of hardcoded 0.0.
    """
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
    """Scan folder, return STRINGS (not Path objects)"""
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
    """Same fix for non-signer version"""
    root = Path(root_dir)
    paths, labels = [], []
    
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for npz in sorted(class_dir.glob('*.npz')):
            paths.append(str(npz)) 
            labels.append(class_dir.name)
    
    return paths, labels

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
    if shuffle: ds = ds.shuffle(len(paths))
    if repeat: ds = ds.repeat()
    
    # Zip mean/std
    mean_ds = tf.data.Dataset.from_tensor_slices([global_mean]).repeat(len(paths))
    std_ds = tf.data.Dataset.from_tensor_slices([global_std]).repeat(len(paths))
    ds = tf.data.Dataset.zip((ds, mean_ds, std_ds))
    
    ds = ds.map(lambda d, m, s: parse_npz(d[0], d[1], m, s), 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

DATASET_PATH = '.'
BASE_DIR = Path(DATASET_PATH)
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Dataset path: {DATA_DIR}')
print(f'Output directory: {OUTPUT_DIR}')

paths, labels = scan_dataset(DATA_DIR)
print(f'Found {len(paths)} samples in {len(set(labels))} classes')

USE_SIGNER_AWARE = True 
K_FOLDS = 5 
CURRENT_FOLD = 0  

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

def build_tf_dataloaders(root_dir, max_len=150, batch_size=32, k_folds=5, current_fold=0, 
                        le_fitted=None):
    paths, labels, signer_ids = scan_dataset_with_signers(root_dir)
    
    le = le_fitted if le_fitted is not None else LabelEncoder().fit(labels)
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

import random

all_paths, all_labels, all_signer_ids = scan_dataset_with_signers(DATA_DIR)
label_encoder_global = LabelEncoder()
label_encoder_global.fit(all_labels)
num_classes_global = len(label_encoder_global.classes_)
print(f"Global classes: {list(label_encoder_global.classes_)}")

unique_signers = sorted(set(all_signer_ids))
rng = random.Random(42)
unique_signers_shuffled = unique_signers.copy()
rng.shuffle(unique_signers_shuffled)

# Sisihkan 1 signer sebagai test set permanen
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

batch_size = int(CONFIG['batch_size'])

test_paths = [p for p, s in zip(all_paths, all_signer_ids) if s == TEST_SIGNER]
test_labels_list = [l for l, s in zip(all_labels, all_signer_ids) if s == TEST_SIGNER]
test_ds = create_tf_dataset(
    test_paths, test_labels_list, label_encoder_global,
    signer_info['global_mean'], signer_info['global_std'],
    batch_size=BATCH_SIZE,
    shuffle=False
)
print(f"Test set size: {len(test_paths)} samples")

# ## 2. Model Architecture


def build_mobile_sign_gru(input_dim, num_classes, max_len, hidden_dim=192, num_layers=2, dropout=0.25, bidirectional=True):
    """
    TensorFlow/Keras implementation of MobileSignGRU with explicit shapes.
    """
    inputs = layers.Input(shape=(max_len, input_dim), name='input')
    x = layers.Dropout(dropout*2)(inputs) 
    
    for i in range(num_layers):
        gru = layers.GRU(
            hidden_dim, 
            return_sequences=(i < num_layers-1),
            dropout=dropout if num_layers > 1 else 0.0,
            recurrent_dropout=0.0,  
            reset_after=True,      
            implementation=1,
            name=f'gru_{i}'
        )
        if bidirectional:
            x = layers.Bidirectional(gru)(x)
        else:
            x = gru(x)
    
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation=None, name='output')(x)
    
    return models.Model(inputs, outputs)

print("TensorFlow MobileSignGRU model defined with explicit input shapes.")

# ## 3. Training


def train_tf_model(train_ds, val_ds, CONFIG, steps_per_epoch, validation_steps):
    """Training loop for TensorFlow model."""
    input_dim = CONFIG['input_dim']
    num_classes = CONFIG['num_classes']
    max_len = CONFIG['max_len']

    model = build_mobile_sign_gru(
        input_dim,
        num_classes,
        max_len,
        CONFIG['hidden_dim'],
        CONFIG['num_layers'],
        CONFIG['dropout'],
        CONFIG['bidirectional']
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=CONFIG['patience'],
            restore_best_weights=True,
            monitor='val_loss'
        )
    ]

    for x, y in train_ds.take(1):
        print(x.shape, y.shape)
    
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=CONFIG['epochs'],
        callbacks=callbacks
    )

    return model, history

print(f"Using TensorFlow version: {tf.__version__}")

input_dim = CONFIG['input_dim']
CONFIG['num_classes'] = num_classes
print(f"Input dimension: {input_dim}")

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

print("Starting TensorFlow training pipeline...")

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

if steps_per_epoch < 0:
    n_train = len([s for s in all_signer_ids if s != TEST_SIGNER and s not in signer_info['val_signers']])
    steps_per_epoch = max(1, n_train // CONFIG['batch_size'])

val_steps = tf.data.experimental.cardinality(val_ds).numpy()

if val_steps < 0:
    val_steps = None

model, history_obj = train_tf_model(train_ds, val_ds, CONFIG, steps_per_epoch, val_steps)

history = history_obj.history
fold_results = [{'history': history, 'accuracy': max(history.get('val_accuracy', [0]))}]
best_fold_idx = 0

print("\nTraining complete. Evaluating on held-out test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy (held-out signer {TEST_SIGNER}): {test_acc:.4f}")


all_preds, all_true = [], []
for x_batch, y_batch in test_ds:
    logits = model(x_batch, training=False)
    all_preds.extend(tf.argmax(logits, axis=1).numpy())
    all_true.extend(y_batch.numpy())
all_preds = np.array(all_preds)
all_true = np.array(all_true)

class_names = list(label_encoder_global.classes_)
eval_results = {
    'accuracy': float((all_preds == all_true).mean()),
    'f1_macro': float(f1_score(all_true, all_preds, average='macro', zero_division=0)),
    'f1_weighted': float(f1_score(all_true, all_preds, average='weighted', zero_division=0)),
    'precision': float(precision_score(all_true, all_preds, average='macro', zero_division=0)),
    'recall': float(recall_score(all_true, all_preds, average='macro', zero_division=0)),
    'confusion_matrix': confusion_matrix(all_true, all_preds),
    'class_report': classification_report(all_true, all_preds, target_names=class_names, output_dict=True, zero_division=0),
    'class_names': class_names,
    'predictions': all_preds,
    'labels': all_true,
}
print(f"Test Accuracy: {eval_results['accuracy']:.4f}  F1-macro: {eval_results['f1_macro']:.4f}")

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

with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump(CONFIG_for_export, f, indent=2)

print(f"Model history and CONFIG saved to {OUTPUT_DIR}")


import matplotlib.pyplot as plt

current_history = fold_results[best_fold_idx].get('history', {})

if not current_history or 'loss' not in current_history:
    print("Warning: No training history found to plot.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(current_history['loss']) + 1)

    axes[0].plot(epochs_range, current_history['loss'], label='Train')
    axes[0].plot(epochs_range, current_history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, current_history['accuracy'], label='Train')
    axes[1].plot(epochs_range, current_history['val_accuracy'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150)
    # plt.show()
    print(f"Training plot saved to {OUTPUT_DIR / 'training_history.png'}")


# ## 3.1 Evaluation Metrics


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    eval_results['confusion_matrix'],
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=eval_results['class_names'],
    yticklabels=eval_results['class_names'],
    ax=ax
)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
# plt.show()

print(f"Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

eval_summary = {
    'accuracy': float(eval_results['accuracy']),
    'f1_macro': float(eval_results['f1_macro']),
    'f1_weighted': float(eval_results['f1_weighted']),
    'precision': float(eval_results['precision']),
    'recall': float(eval_results['recall']),
    'per_class': {cls: eval_results['class_report'][cls] for cls in eval_results['class_names']},
    'confusion_matrix': eval_results['confusion_matrix'].tolist(),
}

with open(OUTPUT_DIR / 'eval_results.json', 'w') as f:
    json.dump(eval_summary, f, indent=2)

np.save(OUTPUT_DIR / 'confusion_matrix.npy', eval_results['confusion_matrix'])
np.save(OUTPUT_DIR / 'predictions.npy', eval_results['predictions'])

print(f"\nEvaluation results saved to {OUTPUT_DIR}")

# ## 4. Benchmarking


param_count = model.count_params()
model_size_mb = sum(w.numpy().nbytes for w in model.weights) / (1024 * 1024)
print(f"Model parameters:  {param_count:,}")
print(f"Model size in RAM: {model_size_mb:.2f} MB")


def benchmark_tf_model(model, input_dim, max_len, n_runs=100):
    """Benchmark TF model inference speed."""
    dummy = tf.random.normal((1, max_len, input_dim))

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy, training=False)
        times.append((time.perf_counter() - start) * 1000)

    return {
        'ms_per_sample': round(float(np.mean(times)), 3),
        'fps': round(1000 / float(np.mean(times)), 1),
    }


print("Benchmarking TF model inference speed...")
bench_tf = benchmark_tf_model(model, input_dim, CONFIG['max_len'])
print(f"  TF Model: {bench_tf['ms_per_sample']:.3f} ms/sample ({bench_tf['fps']:.1f} fps)")

# ## 5. Export to TFLite
saved_model_path = OUTPUT_DIR / 'tf_saved_model'
model.export(saved_model_path) 
print(f"SavedModel exported to {saved_model_path}")

def convert_saved_model_to_tflite(saved_model_dir, output_path, quantization='fp16'):
    """Convert TensorFlow SavedModel to TFLite format with GRU compatibility."""
    import tensorflow as tf

    print(f"Converting to TFLite with quantization='{quantization}'...")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    if quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: [
            [tf.random.normal((1, CONFIG['max_len'], input_dim))]
        ]
    elif quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"TFLite model saved to {output_path} ({size_mb:.2f} MB)")
        return output_path, size_mb
    except Exception as e:
        print(f"TFLite Conversion failed: {e}")
        raise e

tflite_path = OUTPUT_DIR / 'model_int8.tflite'
tflite_file, tflite_size = convert_saved_model_to_tflite(
    saved_model_path,
    tflite_path,
    quantization='fp16'
)

def benchmark_tflite_model(tflite_path, n_runs=100):
    """Benchmark TFLite model inference, auto-detecting required input shape."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    flex_delegate = tf.lite.experimental.load_delegate('libtensorflowlite_flex_delegate.so')
    interpreter.add_delegate(flex_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(f"Detected TFLite input shape: {input_shape}")

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': round(np.mean(times), 3),
        'std_ms': round(np.std(times), 3),
        'fps': round(1000 / np.mean(times), 1),
    }

print("Benchmarking TFLite model...")
tflite_bench = benchmark_tflite_model(tflite_path)
print(f"TFLite inference: {tflite_bench['mean_ms']:.3f} +/- {tflite_bench['std_ms']:.3f} ms ({tflite_bench['fps']:.1f} fps)")

# ## 6. Summary & Download


# Final Summary
print("=" * 60)
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

if USE_SIGNER_AWARE and K_FOLDS > 1:
    print("\nCross-Validation Summary:")
    accuracies = [r['accuracy'] for r in fold_results]
    print(f"   Best Val Accuracy: {max(accuracies):.4f}")

print("\nOutput Files:")
for f in sorted(OUTPUT_DIR.glob('*')):
    if f.is_file():
        size = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name}: {size:.2f} MB")

print('\nOutput files available at /kaggle/working/output:')
for f in sorted(OUTPUT_DIR.glob('*')):
    if f.is_file():
        size = f.stat().st_size / (1024 * 1024)
        print(f'   {f.name}: {size:.2f} MB')
print('\nDownload from the Kaggle notebook Output tab.')