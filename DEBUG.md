# Debugging Report: model_raw.tflite

**Date:** 2026-06-14
**Author:** LLM-assisted debugging session
**Initial accuracy:** ~0.3% (random / collapse)
**Final accuracy:** 94.59% (2743/2900)

## Background

The `model_raw.tflite` wraps a trained GRU classifier with raw-input preprocessing (NaN detection, normalization, mask concat) so that raw MediaPipe landmarks can be fed directly without any external preprocessing. The core TF model achieved 81.7% CV accuracy, but the self-contained TFLite wrapper produced near-random outputs.

## Bug 1: Weight Loading — Unmatched Variables

**Symptoms:** TF core model was accurate (20/20 on first 20 samples) but the rebuilt model inside the wrapper had randomly initialized weights for 6 variables (dense, batch_norm).

**Root cause:** `load_weights_from_savedmodel()` only tried exact variable name matches or a `_1` suffix. The SavedModel had `dense_5/kernel`, `batch_normalization_5/gamma` etc. (suffix `_5` from layer naming during training). The fallback for `_1` didn't match.

**Fix in `src/export.py:load_weights_from_savedmodel()`:**
Added generic `_N` suffix matching — for any unmatched variable `base/rest`, search all SavedModel names for `base_N/rest` where N is a digit.

**Verification:** Before fix: 28/34 matched. After: 34/34 matched.

## Bug 2: Normalization Formula — Normalization Layer Epsilon Mismatch

**Symptoms:** After fixing weight loading, accuracy improved to ~14% but far from expected 81%. TF wrapper predictions differed from TF core on the same input despite same weights.

**Root cause:** The wrapper used `tf.keras.layers.Normalization(mean=μ, variance=σ²)` which normalizes as:
```
y = (x - μ) / √(σ² + 0.001)
```
But training normalizes as:
```
y = (x - μ) / (σ + 1e-8)
```
The `Normalization` layer adds ε=0.001 *before* taking the square root (i.e., to variance), while training adds ε=1e-8 *after* taking the square root (i.e., to std). For low-variance landmark channels (std ~0.01), this difference is 3×:
- Normalization: `(x - μ) / √(0.0001 + 0.001)` = `(x - μ) / 0.033`
- Training: `(x - μ) / (0.01 + 1e-8)` = `(x - μ) / 0.01`

**Fix in `src/export.py:export_selfcontained_tflite()`:**
Replaced `Normalization` layer with a Lambda using the exact formula:
```python
normalize = Lambda(lambda x: (x - tf.constant(m)) / (tf.constant(s) + 1e-8))
```
where `m` and `s` are numpy arrays whose `reshape(1,1,-1)` shape matches the input.

## Bug 3: Padding Contamination — Zeros Normalized Into Non-Zero Values

**Symptoms:** After fixing normalization, accuracy was still ~14%. Most predictions collapsed to a few classes.

**Root cause:** Two issues compounded:

1. **Training pipeline order:** `extract_features_with_mask()` → normalize → concat mask → **then pad** with zeros. Padded positions have value 0 with mask=0.
2. **Wrapper order:** Pad with zeros → normalize all positions. Padded zeros get normalized to `(0 - μ) / σ = -μ/σ` (non-zero!). The GRU receives non-zero values for padded positions.

The model was trained to expect zeros in padded positions, but the wrapper was feeding it `-μ/σ` values — completely different.

**Fix:** Two changes:

*a) `test.py:64` — Pad with NaN, not zeros:*
```python
input_arr = np.full((1, max_len, raw_dim), np.nan, dtype=np.float32)
input_arr[0, :T] = raw_feat[:T]
```

*b) `src/export.py:export_selfcontained_tflite()` — Zero-out after normalization:*
```python
# Step 1: valid_mask = 1 for valid landmarks, 0 for NaN/padding
valid_mask = Lambda(lambda x: tf.cast(tf.logical_not(tf.math.is_nan(x)), tf.float32))(raw_input)
# Step 2: NaN→0, then normalise ALL positions
normalized = Lambda(lambda x: (tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) - tf.constant(m)) / (tf.constant(s) + 1e-8))(raw_input)
# Step 3: zero-out NaN/padding positions AFTER normalisation
masked_normalized = Lambda(lambda x: x[0] * x[1])([normalized, valid_mask])
# Step 4: concat features + mask
preprocessed = Concatenate(axis=-1)([masked_normalized, valid_mask])
```

## Timeline

| Step | Accuracy | What changed |
|------|----------|-------------|
| Initial | ~0.3% | model_raw.tflite with Lambda closures + tf.equal(x,x) for NaN detection |
| Custom layer fix | ~0.3% | Used `RawPreprocessor` layer with `add_weight` (TFLite can't read resource vars) |
| tf.constant fix | ~0.3% | Embedded mean/std as `tf.constant` (still broken in TFLite) |
| Normalization layer | ~14% | Used `tf.keras.layers.Normalization` with precomputed mean/variance |
| Weight loading fix | ~14% | Matched all 34/34 weights (uncovered Bug 2 & 3) |
| Normalization formula fix | ~14% | Replaced Normalization layer with Lambda matching training formula (uncovered Bug 3) |
| NaN padding + zero-out after norm | **94.59%** | Complete fix |

## Key Lessons

1. **TFLite hates resource variables.** Custom preprocessing layers with `add_weight()` won't serialize correctly. Use `tf.constant()` inside `Lambda` functions instead.

2. **Normalization epsilon matters.** `Normalization(mean, variance)` adds epsilon BEFORE sqrt. Training's `(x - μ) / (σ + ε)` adds epsilon AFTER sqrt. For low-variance features, this is catastrophic.

3. **Normalize first, pad second.** The training pipeline normalizes actual features, then pads with zeros. A self-contained wrapper must preserve this order: normalize all, then zero-out padded/NaN positions.

4. **Pad with NaN, not zeros.** Use `np.full(..., np.nan)` for padding so the wrapper can distinguish padding from valid zero-valued landmarks.

5. **SavedModel variable names are fragile.** Layer names get `_N` suffixes depending on construction order. Weight loading must try all suffixes, not just `_1`.
