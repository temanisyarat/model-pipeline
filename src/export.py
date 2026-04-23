import time
import numpy as np
import tensorflow as tf
from pathlib import Path


def convert_saved_model_to_tflite(saved_model_dir, output_path, CONFIG, input_dim, quantization='fp16'):
    """Convert TensorFlow SavedModel to TFLite format."""
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


def benchmark_tflite_model(tflite_path, n_runs=100):
    """Benchmark TFLite model inference."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    flex_delegate = tf.lite.experimental.load_delegate('libtensorflowlite_flex_delegate.so')
    interpreter.add_delegate(flex_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(f"Detected TFLite input shape: {input_shape}")

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

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