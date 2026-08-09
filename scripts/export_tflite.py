"""Export the ASL classifier to an optimized TensorFlow Lite model.

Usage:
    python -m scripts.export_tflite
    python -m scripts.export_tflite --model-path path/to/model.keras
"""

import argparse
import os

from utils import KERAS_MODEL_PATH, TFLITE_MODEL_PATH


def export_tflite(model_path=KERAS_MODEL_PATH, output_path=TFLITE_MODEL_PATH,
                  optimize=True):
    """Convert a Keras or legacy SavedModel artifact to TensorFlow Lite.

    Dynamic-range quantization is used by default. It quantizes model weights
    while keeping float32 inputs and outputs, so the application can use the
    exported model without changing its landmark feature pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Import TensorFlow lazily so `--help` and module inspection work without
    # importing the heavyweight ML runtime.
    import tensorflow as tf

    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    if not tflite_model:
        raise RuntimeError("TensorFlow Lite conversion returned an empty model")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Avoid replacing a previously working artifact if conversion or writing
    # is interrupted midway through.
    temporary_path = f"{output_path}.tmp"
    try:
        with open(temporary_path, 'wb') as file:
            file.write(tflite_model)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"TensorFlow Lite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
    return output_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Export the ASL classifier as an optimized TensorFlow Lite model'
    )
    parser.add_argument(
        '--model-path', default=KERAS_MODEL_PATH,
        help=f'Input model path (default: {KERAS_MODEL_PATH})'
    )
    parser.add_argument(
        '--output', default=TFLITE_MODEL_PATH,
        help=f'Output model path (default: {TFLITE_MODEL_PATH})'
    )
    parser.add_argument(
        '--no-quantization', action='store_true',
        help='Disable dynamic-range weight quantization'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_tflite(
        model_path=args.model_path,
        output_path=args.output,
        optimize=not args.no_quantization,
    )
