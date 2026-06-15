# AGENTS.md — TemanIsyarat BISINDO Model

## Project Overview
BISINDO sign language word recognition from MediaPipe landmarks. 14 signers, 20 classes, ~2900 samples. GRU-based classifier with self-contained TFLite export.

## Key Files
- `main.py` — Full pipeline: data → leave-one-out CV → export best model
- `test.py` — Eval `model_raw.tflite` on all samples; pads with `np.nan`
- `src/config.py` — All hyperparams (max_len=110, hidden_dim=128, dropout=0.5, l2=3e-3)
- `src/model.py` — `build_mobile_sign_gru()`: Conv1D×2 → BiGRU×2 → Attention → Dense
- `src/data.py` — `extract_features_with_mask()`: NaN→0, return feat+mask. `parse_npz()`: normalize FIRST, then pad
- `src/export.py` — `export_selfcontained_tflite()`: builds wrapper with NaN handling + normalization matching training formula exactly: `(x - mean) / (std + 1e-8)`, then `normalized × valid_mask` to zero-out padding

## Critical: Normalization Order
Training normalizes BEFORE padding. The TFLite wrapper:
1. Detect NaN → mask (0=invalid/padding, 1=valid)
2. NaN→0, normalize all positions
3. `normalized × mask` — zeros-out padding & missing landmarks
4. Concat `[masked_normalized | mask]` → model

## Critical: Weight Loading Fallback
SavedModel layer names have `_N` suffix (e.g., `dense_5/kernel`). `load_weights_from_savedmodel()` in export.py tries all `_N` variants.

## Data Layout
`data/{signer}/{class}/*.npz`. Features: `pose(T,9,4)[:,:,:3]` (27) + `hands(T,2,21,3)` (126) = raw_dim=153. Model input dim=306 (153 feat + 153 mask).

## Output Models
- `output/model.tflite` — Standard (expects preprocessed 306-dim input)
- `output/model_raw.tflite` — Self-contained (expects raw 153-dim with NaN padding)

## Known Pitfalls
- TFLite `Normalization` layer uses `√(variance + 0.001)` — does NOT match training formula. Use Lambda with `tf.constant`.
- Do NOT use `add_weight` in custom preprocessing layers — TFLite can't read resource variables. Use `tf.constant` inside Lambda functions.
- Padding must be NaN (not 0) so wrapper can distinguish real zeros from padding.

## Commands
```bash
python main.py           # Full CV train + export
python test.py           # Eval model_raw.tflite
source .venv/bin/activate && python3 {script}
```
