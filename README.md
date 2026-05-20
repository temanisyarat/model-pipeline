# TemanIsyarat — BISINDO Recognition Model Pipeline

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.21-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/keras-3.14-red)](https://keras.io/)

A deep learning pipeline for **BISINDO (Bahasa Isyarat Indonesia / Indonesian Sign Language)** word recognition. The system takes MediaPipe pose and hand landmark sequences extracted from video and classifies them into one of 20 Indonesian sign language words using a GRU-based recurrent neural network with temporal attention.

## Overview

This project provides an end-to-end ML pipeline:

- **Data ingestion** — scans `.npz` files organized by signer identity and sign class
- **Signer-independent evaluation** — held-out signer testing with k-fold cross-validation
- **Feature processing** — masking-aware normalization, sequence padding/truncation
- **Model** — `MobileSignGRU`: a compact GRU architecture with 1D convolutions and temporal attention (~1.3M params, ~5 MB FP32)
- **Training** — data augmentation (noise, channel dropout, time masking), label smoothing, gradient clipping, early stopping, learning rate scheduling
- **Evaluation** — accuracy, F1-score (macro/micro), precision, recall, confusion matrix
- **Export** — TFLite conversion with FP16 and INT8 quantization for mobile/edge deployment
- **Benchmarking** — inference speed comparison between TensorFlow and TFLite models

## Dataset

The dataset contains **1,800 `.npz` files** across **9 signers** and **20 sign classes**:

| Signers                                                                       | Classes                                                                                                                                                                        |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `farras`, `fredi`, `ian`, `ivan`, `willi`, `hany`, `mutia`, `saidah`, `kevin` | `aku`, `apel`, `ayah`, `besok`, `buku`, `dia`, `dua`, `hari ini`, `ibu`, `kamu`, `kuning`, `maaf`, `merah`, `nama`, `pisang`, `salam`, `satu`, `teman`, `terima kasih`, `tiga` |

Each `.npz` file contains MediaPipe-extracted landmarks:

| Key     | Shape           | Description                                |
| ------- | --------------- | ------------------------------------------ |
| `pose`  | `(T, 9, 4)`     | 9 pose keypoints × `[x, y, z, visibility]` |
| `hands` | `(T, 2, 21, 3)` | 2 hands × 21 keypoints × `[x, y, z]`       |

Each signer provides 10 recordings per class, each augmented into 8 variants (original, fast, slow, horizontal flip, and combinations).

## Model Architecture

```
Input (125, 306)
  ├── Mask Concatenation (feat=153, mask=153)
  ├── SpatialDropout1D (0.2)
  ├── Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
  ├── Bidirectional(GRU(192)) × 2
  ├── TemporalAttention
  ├── Dropout(0.3) → Dense(192, ReLU) → BatchNorm → Dropout(0.3)
  └── Dense(20, logits)
```

**Parameters:** 1,296,853 (~1.3M)  
**Model size:** ~4.95 MB (FP32), ~2.6 MB (TFLite FP16)

## Installation

### Prerequisites

- Python 3.12
- TensorFlow 2.21+

### Setup

```bash
git clone https://github.com/williamu04/temanisyarat-model-pipeline.git
cd temanisyarat-model-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the full pipeline (data loading → training → evaluation → export):

```bash
python main.py
```

The pipeline will:

1. Scan and load `.npz` files from the `data/` directory
2. Split data using signer-independent k-fold cross-validation
3. Train the `MobileSignGRU` model with data augmentation
4. Evaluate on a held-out signer
5. Export the model to TFLite (FP16 quantized)
6. Benchmark inference speed

### Configuration

All hyperparameters are centralized in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `max_len` | 125 | Maximum sequence frames |
| `batch_size` | 64 | Training batch size |
| `hidden_dim` | 192 | GRU hidden dimension |
| `num_layers` | 2 | Number of GRU layers |
| `dropout` | 0.3 | Dropout rate |
| `learning_rate` | 3e-4 | Initial learning rate |
| `epochs` | 100 | Maximum training epochs |
| `patience` | 20 | Early stopping patience |
| `conv_filters` | [128, 128] | 1D Conv filter counts |

### Output

All artifacts are saved to the `output/` directory:

| File | Description |
|---|---|
| `tf_saved_model/` | TensorFlow SavedModel |
| `model_fp16.tflite` | FP16 quantized TFLite model |
| `model_int8.tflite` | INT8 quantized TFLite model |
| `eval_results.json` | Evaluation metrics |
| `confusion_matrix.png` | Confusion matrix visualization |
| `training_history.png` | Training & validation curves |
| `config.json` | Frozen training configuration |

## Evaluation

The model is evaluated using **signer-independent** splitting — the test signer is completely unseen during training. Metrics reported:

- Accuracy
- F1-score (macro and micro)
- Precision
- Recall
- Confusion matrix

## Project Structure

```
├── main.py              # Pipeline orchestrator
├── requirements.txt     # Python dependencies
├── src/
│   ├── config.py        # Configuration & hyperparameters
│   ├── data.py          # Data loading, preprocessing, augmentation
│   ├── model.py         # MobileSignGRU architecture
│   ├── train.py         # Training loop & callbacks
│   ├── evaluate.py      # Evaluation metrics & visualization
│   └── export.py        # TFLite conversion & benchmarking
├── data/                # Dataset (.npz files per signer/class)
└── output/              # Training artifacts & evaluation results
```

## Development

### Code Style

```bash
pip install ruff
ruff check src/ main.py
```

### Adding New Classes

1. Add recordings to `data/<signer>/<new_class>/` as `.npz` files
2. Add the class label to the `CLASSES` list in [`src/config.py`](src/config.py)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{temanisyarat_model,
  title = {TemanIsyarat: BISINDO Recognition Model Pipeline},
  author = {Temanisyarat Team and contributors},
  url = {https://github.com/temanisyarat/model-pipeline},
  year = {2026}
}
```

## Acknowledgments

Built with [TensorFlow](https://www.tensorflow.org/) and [MediaPipe](https://mediapipe.dev/) for Indonesian Sign Language (BISINDO) recognition.
