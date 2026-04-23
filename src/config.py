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


CONFIG = {
    'max_len': 100,
    'batch_size': 128,
    'input_dim': 252,
    'num_classes': 5,
    'hidden_dim': 192,
    'num_layers': 2,
    'dropout': 0.25,
    'bidirectional': True,
    'learning_rate': 1e-3,
    'epochs': 20,
    'patience': 12,
    'label_smoothing': 0.1,
}

BATCH_SIZE = int(CONFIG['batch_size'])
MAX_LEN = int(CONFIG['max_len'])
INPUT_DIM = int(CONFIG['input_dim'])

CLASSES = ['aku', 'apa', 'dia', 'kamu', 'siapa']

DATASET_PATH = '.'
BASE_DIR = Path(DATASET_PATH)
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded.")
print(f"Classes: {CLASSES}")
print(f'Dataset path: {DATA_DIR}')
print(f'Output directory: {OUTPUT_DIR}')