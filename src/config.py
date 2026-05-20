from pathlib import Path

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")


CONFIG = {
    "max_len": 125,
    "batch_size": 64,
    "input_dim": 153,
    "num_classes": 20,
    "hidden_dim": 192,
    "num_layers": 2,
    "dropout": 0.3,
    "recurrent_dropout": 0.2,
    "bidirectional": True,
    "learning_rate": 3e-4,
    "epochs": 100,
    "patience": 20,
    "label_smoothing": 0.1,
    "l2_reg": 1e-3,
    "gradient_clip_norm": 1.0,
    "conv_filters": [128, 128],
    "conv_kernel_size": 5,
    "use_mask_concat": True,
    "spatial_dropout": 0.2,
}

BATCH_SIZE = int(CONFIG["batch_size"])
MAX_LEN = int(CONFIG["max_len"])
INPUT_DIM = int(CONFIG["input_dim"])
MODEL_INPUT_DIM = int(
    CONFIG["input_dim"] * (2 if CONFIG.get("use_mask_concat", False) else 1)
)

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

DATASET_PATH = "."
BASE_DIR = Path(DATASET_PATH)
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded.")
print(f"Classes: {CLASSES}")
print(f"Dataset path: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

