"""Tests for TensorFlow Lite export and runtime backend selection."""

import numpy as np
import pytest

from model_runtime import load_classifier
from scripts.export_tflite import export_tflite


tf = pytest.importorskip("tensorflow")


@pytest.fixture
def model_artifacts(tmp_path):
    """Create a tiny compatible Keras model and export its TFLite artifact."""
    model_path = tmp_path / "classifier.keras"
    tflite_path = tmp_path / "classifier.tflite"

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.save(str(model_path))
    export_tflite(str(model_path), str(tflite_path))

    return model_path, tflite_path


def test_export_creates_usable_tflite_model(model_artifacts):
    """The exporter should create a non-empty model with the expected input."""
    _, tflite_path = model_artifacts

    assert tflite_path.exists()
    assert tflite_path.stat().st_size > 0

    classifier = load_classifier(
        expected_output_size=2,
        tflite_path=str(tflite_path),
        keras_path='',
        legacy_path='',
    )
    prediction = classifier.predict(np.zeros((1, 42), dtype=np.float32))

    assert classifier.backend == 'tflite'
    assert prediction.shape == (1, 2)
    assert np.isclose(np.sum(prediction), 1.0, atol=0.01)


def test_tflite_is_preferred_over_keras(model_artifacts):
    """The runtime should select TFLite when both artifacts are available."""
    model_path, tflite_path = model_artifacts

    classifier = load_classifier(
        expected_output_size=2,
        tflite_path=str(tflite_path),
        keras_path=str(model_path),
        legacy_path='',
    )

    assert classifier.backend == 'tflite'


def test_runtime_falls_back_to_keras(model_artifacts):
    """A missing TFLite artifact should use the Keras model."""
    model_path, _ = model_artifacts

    classifier = load_classifier(
        expected_output_size=2,
        tflite_path='',
        keras_path=str(model_path),
        legacy_path='',
    )

    prediction = classifier.predict(np.zeros((1, 42), dtype=np.float32))
    assert classifier.backend == 'keras'
    assert prediction.shape == (1, 2)


def test_runtime_falls_back_when_tflite_is_invalid(model_artifacts):
    """A corrupt TFLite artifact should not prevent Keras fallback."""
    model_path, tflite_path = model_artifacts
    tflite_path.write_bytes(b"not a TensorFlow Lite model")

    classifier = load_classifier(
        expected_output_size=2,
        tflite_path=str(tflite_path),
        keras_path=str(model_path),
        legacy_path='',
    )

    assert classifier.backend == 'keras'
