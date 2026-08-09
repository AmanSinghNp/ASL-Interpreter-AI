"""Runtime adapters for Keras and TensorFlow Lite ASL classifiers."""

import os

import numpy as np

from utils import (
    FEATURE_COUNT,
    KERAS_MODEL_PATH,
    MODEL_PATH,
    TFLITE_MODEL_PATH,
)


class KerasClassifier:
    """Small adapter exposing a common prediction interface."""

    backend = 'keras'

    def __init__(self, model):
        self.model = model
        self.input_shape = tuple(model.input_shape)
        self.output_shape = tuple(model.output_shape)

    def predict(self, features):
        """Return model probabilities for a `(batch, 42)` feature array."""
        features = _validate_features(features)
        return np.asarray(self.model.predict(features, verbose=0), dtype=np.float32)


class SavedModelClassifier:
    """Adapter for legacy TensorFlow SavedModel directories."""

    backend = 'savedmodel'

    def __init__(self, signature):
        self.signature = signature
        input_signature = signature.structured_input_signature[1]
        if len(input_signature) != 1:
            raise ValueError("SavedModel must have exactly one input tensor")

        self.input_name = next(iter(input_signature))
        input_spec = input_signature[self.input_name]
        input_shape = tuple(
            -1 if value is None else int(value)
            for value in input_spec.shape
        )
        if len(input_shape) != 2 or input_shape[-1] != FEATURE_COUNT:
            raise ValueError(
                f"SavedModel must accept ({FEATURE_COUNT},) features; "
                f"got input shape {input_shape}"
            )
        self.input_shape = input_shape

        output_signature = signature.structured_outputs
        if len(output_signature) != 1:
            raise ValueError("SavedModel must have exactly one output tensor")
        output_spec = next(iter(output_signature.values()))
        self.output_shape = tuple(
            -1 if value is None else int(value)
            for value in output_spec.shape
        )

    def predict(self, features):
        """Return model probabilities for a `(batch, 42)` feature array."""
        features = _validate_features(features)
        import tensorflow as tf

        output = self.signature(
            **{self.input_name: tf.convert_to_tensor(features)}
        )
        return np.asarray(next(iter(output.values())).numpy(), dtype=np.float32)


class TFLiteClassifier:
    """TensorFlow Lite classifier adapter with quantized I/O support."""

    backend = 'tflite'

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()[0]
        self.output_details = interpreter.get_output_details()[0]
        self.input_shape = tuple(int(value) for value in self.input_details['shape'])
        self.output_shape = tuple(
            int(value) for value in self.output_details['shape']
        )

        if len(self.input_shape) != 2 or self.input_shape[-1] != FEATURE_COUNT:
            raise ValueError(
                f"TFLite model must accept ({FEATURE_COUNT},) features; "
                f"got input shape {self.input_shape}"
            )

    def predict(self, features):
        """Return model probabilities for a `(1, 42)` feature array."""
        features = _validate_features(features)
        if tuple(features.shape) != self.input_shape:
            raise ValueError(
                f"TFLite model expects input shape {self.input_shape}; "
                f"got {tuple(features.shape)}"
            )

        input_tensor = _quantize_tensor(features, self.input_details)
        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details['index'])
        return _dequantize_tensor(output, self.output_details)


def _validate_features(features):
    """Normalize and validate the feature array shared by both backends."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2 or features.shape[-1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected features with shape (batch, {FEATURE_COUNT}); "
            f"got {features.shape}"
        )
    return features


def _quantization_params(tensor_details):
    """Read TFLite quantization parameters from tensor metadata."""
    scale, zero_point = tensor_details.get('quantization', (0.0, 0))
    return float(scale), int(zero_point)


def _quantize_tensor(values, tensor_details):
    """Convert float values to a quantized TFLite tensor when required."""
    dtype = tensor_details['dtype']
    scale, zero_point = _quantization_params(tensor_details)
    if not scale or np.issubdtype(dtype, np.floating):
        return values.astype(dtype)

    limits = np.iinfo(dtype)
    quantized = np.round(values / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize_tensor(values, tensor_details):
    """Convert a TFLite output tensor to float32 probabilities."""
    scale, zero_point = _quantization_params(tensor_details)
    if not scale or np.issubdtype(values.dtype, np.floating):
        return np.asarray(values, dtype=np.float32)
    return (values.astype(np.float32) - zero_point) * scale


def _load_tflite(path):
    """Load and allocate a TensorFlow Lite interpreter."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return TFLiteClassifier(interpreter)


def _load_saved_model(path):
    """Load a legacy SavedModel using the Keras-3-compatible TensorFlow API."""
    import tensorflow as tf

    loaded = tf.saved_model.load(path)
    signatures = loaded.signatures
    if not signatures:
        raise ValueError("SavedModel has no callable signatures")

    signature = signatures.get('serving_default') or next(iter(signatures.values()))
    return SavedModelClassifier(signature)


def _load_keras(path):
    """Load a Keras model or legacy SavedModel directory."""
    if os.path.isdir(path):
        return _load_saved_model(path)

    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    input_shape = tuple(model.input_shape)
    if len(input_shape) != 2 or input_shape[-1] != FEATURE_COUNT:
        raise ValueError(
            f"Keras model must accept ({FEATURE_COUNT},) features; "
            f"got input shape {input_shape}"
        )
    return KerasClassifier(model)


def _validate_output_size(classifier, expected_output_size):
    """Validate that the model has one output probability per class."""
    output_shape = classifier.output_shape

    if len(output_shape) != 2 or output_shape[-1] != expected_output_size:
        raise ValueError(
            f"Model output size {output_shape} does not match "
            f"{expected_output_size} loaded classes"
        )


def load_classifier(expected_output_size, tflite_path=TFLITE_MODEL_PATH,
                    keras_path=KERAS_MODEL_PATH, legacy_path=MODEL_PATH):
    """Load the fastest available classifier with graceful fallback.

    TensorFlow Lite is preferred when available. A `.keras` artifact is the
    primary fallback, followed by the legacy SavedModel directory used by
    older versions of this project.
    """
    errors = []
    candidates = [
        ('tflite', tflite_path, _load_tflite),
        ('keras', keras_path, _load_keras),
        ('legacy keras', legacy_path, _load_keras),
    ]

    for name, path, loader in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            classifier = loader(path)
            _validate_output_size(classifier, expected_output_size)
            return classifier
        except Exception as error:
            errors.append(f"{name} ({path}): {error}")

    searched_paths = ', '.join(
        path for _, path, _ in candidates if path
    )
    message = f"No usable ASL model found. Checked: {searched_paths}."
    if errors:
        message += " Errors: " + ' | '.join(errors)
    raise RuntimeError(message)
