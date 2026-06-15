# TemanIsyarat — BISINDO Recognition Model Pipeline

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.21-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/keras-3.14-red)](https://keras.io/)

A deep learning pipeline for **BISINDO (Bahasa Isyarat Indonesia / Indonesian Sign Language)** word recognition. The system takes MediaPipe pose and hand landmark sequences extracted from video and classifies them into one of 20 Indonesian sign language words using a GRU-based recurrent neural network with temporal attention.

## Overview

- **Data ingestion** — scans `.npz` files organized by signer identity and sign class
- **Signer-independent evaluation** — leave-one-signer-out cross-validation (14 signers)
- **Feature processing** — NaN-aware normalization with masking, sequence padding/truncation
- **Model** — `MobileSignGRU`: compact GRU with 1D convolutions + temporal attention (~712K params, ~2.7 MB FP32)
- **Training** — aggressive regularization (dropout 0.5, l2 3e-3, label smoothing 0.2), augmentation (noise, channel drop, time masking), gradient clipping, early stopping, LR scheduling
- **Evaluation** — accuracy, F1, precision, recall, confusion matrix per signer
- **Export** — self-contained `model_raw.tflite` that accepts raw (110, 153) MediaPipe landmarks with NaNs — no external preprocessing needed

## Dataset

**2,900 `.npz` files** across **14 signers** and **20 sign classes**:

| Signers | Classes |
|---|---|
| `farras`, `farras1`, `fredi`, `fredi1`, `ian`, `ian1`, `ivan`, `ivan1`, `willi`, `willi1`, `hany`, `mutia`, `saidah`, `kevin` | `aku`, `apel`, `ayah`, `besok`, `buku`, `dia`, `dua`, `hari-ini`, `ibu`, `kamu`, `kuning`, `maaf`, `merah`, `nama`, `pisang`, `salam`, `satu`, `teman`, `terima-kasih`, `tiga` |

Each `.npz` contains MediaPipe-extracted landmarks:

| Key | Shape | Description |
|---|---|---|
| `pose` | `(T, 9, 4)` | 9 pose keypoints × `[x, y, z, visibility]` |
| `hands` | `(T, 2, 21, 3)` | 2 hands × 21 keypoints × `[x, y, z]` |

Each signer provides ~10 recordings per class, each augmented into 8 variants (original, fast, slow, h-flip, and combinations).

## Model Architecture

```
Input (110, 306)  ← [153 normalized feats | 153 NaN-mask]
  ├── Feature masking (feat × mask)
  ├── SpatialDropout1D (0.2)
  ├── Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Bidirectional(GRU(128)) × 2
  ├── TemporalAttention
  ├── Dropout(0.5) → Dense(128, ReLU, l2=3e-3) → BatchNorm → Dropout(0.5)
  └── Dense(20, logits)
```

**Parameters:** 711,957 (~712K)
**Model size:** ~2.72 MB (FP32), ~2.77 MB (TFLite self-contained)

## Performance

- **Cross-validation (14 folds):** 81.7% ± 9.7% mean accuracy
- **Best fold:** ivan — 96.5%
- **Self-contained TFLite (all 2900 samples):** 94.6% accuracy

## Installation

### Prerequisites

- Python 3.12
- TensorFlow 2.21+

### Setup

```bash
git clone https://github.com/temanisyarat/model.git
cd temanisyarat/model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python main.py
```

Runs leave-one-signer-out CV over all 14 signers, saves best model to `output/tf_saved_model/`.

### Evaluate the self-contained TFLite model

```bash
python test.py
```

Evaluates `output/model_raw.tflite` on all 2900 samples, produces accuracy, confusion matrix, and per-signer breakdown.

### Configuration

All hyperparameters in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `max_len` | 110 | Max sequence frames |
| `batch_size` | 64 | Training batch size |
| `hidden_dim` | 128 | GRU hidden dimension |
| `num_layers` | 2 | Number of GRU layers |
| `dropout` | 0.5 | Dropout rate |
| `recurrent_dropout` | 0.3 | Recurrent dropout |
| `learning_rate` | 3e-4 | Initial learning rate |
| `epochs` | 100 | Max epochs |
| `patience` | 15 | Early stopping patience |
| `l2_reg` | 3e-3 | L2 weight decay |
| `label_smoothing` | 0.2 | Label smoothing factor |
| `conv_filters` | [128, 128] | Conv filter counts |

### Output artifacts (`output/`)

| File | Description |
|---|---|
| `tf_saved_model/` | Best-fold TensorFlow SavedModel |
| `model.tflite` | Standard TFLite (preprocessed input) |
| `model_raw.tflite` | Self-contained TFLite (raw landmarks in, logits out) |
| `config.json` | Frozen training configuration & CV results |
| `test_results.json` | Final evaluation metrics |
| `confusion_matrix.png` | Training confusion matrix |
| `test_confusion_matrix.png` | TFLite evaluation confusion matrix |
| `training_history.png` | Training & validation curves |
| `global_mean.npy` | Normalization mean (baked into model_raw.tflite) |
| `global_std.npy` | Normalization std (baked into model_raw.tflite) |

## Project Structure

```
├── main.py              # Pipeline orchestrator (CV train + export)
├── test.py              # Self-contained TFLite evaluation
├── requirements.txt     # Python dependencies
├── src/
│   ├── config.py        # Configuration & hyperparameters
│   ├── data.py          # Data loading, preprocessing, augmentation
│   ├── model.py         # MobileSignGRU architecture
│   ├── train.py         # Training loop & callbacks
│   ├── evaluate.py      # Evaluation metrics & visualization
│   └── export.py        # TFLite conversion & benchmarking
├── data/                # Dataset (14 signers × 20 classes)
├── archive/             # Raw data before merging
└── output/              # Training artifacts & evaluation results
```

## Model Export: model_raw.tflite

The `model_raw.tflite` file is a **self-contained** model that:

1. Accepts raw `(110, 153)` float32 landmarks (with NaNs for undetected keypoints)
2. Internally detects NaNs, replaces with 0, normalizes using precomputed mean/std
3. Concatenates NaN-mask as extra channels
4. Zeroes-out padded positions after normalization
5. Runs the core GRU classifier
6. Outputs 20-class logits

No external preprocessing is needed — perfect for mobile/edge deployment.

## Citation

```bibtex
@software{temanisyarat_model,
  title = {TemanIsyarat: BISINDO Recognition Model Pipeline},
  author = {Temanisyarat Team},
  url = {https://github.com/temanisyarat/model},
  year = {2026}
}
```

## License

MIT
