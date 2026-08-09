"""Unit tests for model loading and inference."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_runtime import load_classifier
from utils import KERAS_MODEL_PATH, MODEL_PATH, load_classes

MODEL_FOR_TESTS = KERAS_MODEL_PATH if os.path.exists(KERAS_MODEL_PATH) else MODEL_PATH
EXPECTED_CLASSES = load_classes()


def load_test_classifier():
    """Load a generated Keras or legacy SavedModel artifact for tests."""
    return load_classifier(
        expected_output_size=len(EXPECTED_CLASSES),
        tflite_path='',
        keras_path=KERAS_MODEL_PATH if os.path.exists(KERAS_MODEL_PATH) else '',
        legacy_path=MODEL_PATH if os.path.exists(MODEL_PATH) else '',
    )


class TestModelPath:
    """Tests for model path configuration."""

    def test_model_path_defined(self):
        """MODEL_PATH should be defined."""
        assert MODEL_PATH is not None
        assert len(MODEL_PATH) > 0

    def test_model_path_format(self):
        """MODEL_PATH should point to expected location."""
        assert 'saved_model' in MODEL_PATH
        assert 'asl_model' in MODEL_PATH


@pytest.mark.skipif(
    not os.path.exists(MODEL_FOR_TESTS),
    reason="Model not found - run train_model.py first"
)
class TestModelLoading:
    """Tests for loading the trained model through the runtime adapter."""

    @pytest.fixture(scope="class")
    def loaded_classifier(self):
        """Load the generated model once for all tests in this class."""
        return load_test_classifier()

    def test_model_loads_successfully(self, loaded_classifier):
        """Model should load without errors."""
        assert loaded_classifier is not None

    def test_model_has_correct_output_shape(self, loaded_classifier):
        """Model output should match the saved class labels."""
        assert loaded_classifier.output_shape[-1] == len(EXPECTED_CLASSES)

    def test_model_has_correct_input_shape(self, loaded_classifier):
        """Model should expect 42 features."""
        assert loaded_classifier.input_shape[-1] == 42

    def test_model_prediction_shape(self, loaded_classifier):
        """Prediction should return probabilities for each class."""
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_classifier.predict(dummy_input)

        assert prediction.shape == (1, len(EXPECTED_CLASSES))

    def test_prediction_is_probability_distribution(self, loaded_classifier):
        """Predictions should sum to 1 (valid probability distribution)."""
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_classifier.predict(dummy_input)

        assert abs(np.sum(prediction) - 1.0) < 0.01

    def test_prediction_values_in_range(self, loaded_classifier):
        """All prediction values should be between 0 and 1."""
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_classifier.predict(dummy_input)

        assert np.all(prediction >= 0)
        assert np.all(prediction <= 1)

    def test_batch_prediction(self, loaded_classifier):
        """Model should handle batch predictions."""
        batch_size = 5
        dummy_input = np.random.randn(batch_size, 42).astype(np.float32)
        prediction = loaded_classifier.predict(dummy_input)

        assert prediction.shape == (batch_size, len(EXPECTED_CLASSES))


class TestModelIntegration:
    """Integration tests combining model with normalization."""

    @pytest.mark.skipif(
        not os.path.exists(MODEL_FOR_TESTS),
        reason="Model not found - run train_model.py first"
    )
    def test_full_pipeline(self, centered_landmarks):
        """Test complete pipeline from landmarks to prediction."""
        from utils import normalize_landmarks

        classifier = load_test_classifier()
        features = normalize_landmarks(centered_landmarks)
        assert features is not None

        prediction = classifier.predict(features)

        assert prediction.shape == (1, len(EXPECTED_CLASSES))
        assert abs(np.sum(prediction) - 1.0) < 0.01

        predicted_idx = np.argmax(prediction)
        assert 0 <= predicted_idx < len(EXPECTED_CLASSES)

        confidence = np.max(prediction)
        assert 0 <= confidence <= 1
