"""
Unit tests for model loading and inference.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import MODEL_PATH, ASL_LETTERS


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
    not os.path.exists(MODEL_PATH),
    reason="Model not found - run train_model.py first"
)
class TestModelLoading:
    """Tests for loading the trained model."""
    
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load model once for all tests in this class."""
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    
    def test_model_loads_successfully(self, loaded_model):
        """Model should load without errors."""
        assert loaded_model is not None
    
    def test_model_has_correct_output_shape(self, loaded_model):
        """Model output should match number of ASL letters."""
        output_shape = loaded_model.output_shape
        # Output shape should be (None, 24) for 24 classes
        assert output_shape[-1] == len(ASL_LETTERS)
    
    def test_model_has_correct_input_shape(self, loaded_model):
        """Model input should expect 42 features."""
        input_shape = loaded_model.input_shape
        # Input shape should be (None, 42) for 42 features
        assert input_shape[-1] == 42
    
    def test_model_prediction_shape(self, loaded_model):
        """Prediction should return probabilities for each class."""
        # Create dummy input with shape (1, 42)
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (1, len(ASL_LETTERS))
    
    def test_prediction_is_probability_distribution(self, loaded_model):
        """Predictions should sum to 1 (valid probability distribution)."""
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_model.predict(dummy_input, verbose=0)
        
        # Should sum to approximately 1
        assert abs(np.sum(prediction) - 1.0) < 0.01
    
    def test_prediction_values_in_range(self, loaded_model):
        """All prediction values should be between 0 and 1."""
        dummy_input = np.random.randn(1, 42).astype(np.float32)
        prediction = loaded_model.predict(dummy_input, verbose=0)
        
        assert np.all(prediction >= 0)
        assert np.all(prediction <= 1)
    
    def test_batch_prediction(self, loaded_model):
        """Model should handle batch predictions."""
        batch_size = 5
        dummy_input = np.random.randn(batch_size, 42).astype(np.float32)
        prediction = loaded_model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (batch_size, len(ASL_LETTERS))


class TestModelIntegration:
    """Integration tests combining model with normalization."""
    
    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Model not found - run train_model.py first"
    )
    def test_full_pipeline(self, centered_landmarks):
        """Test complete pipeline from landmarks to prediction."""
        import tensorflow as tf
        from utils import normalize_landmarks
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Normalize landmarks
        features = normalize_landmarks(centered_landmarks)
        assert features is not None
        
        # Get prediction
        prediction = model.predict(features, verbose=0)
        
        # Should get valid prediction
        assert prediction.shape == (1, len(ASL_LETTERS))
        assert abs(np.sum(prediction) - 1.0) < 0.01
        
        # Should be able to get predicted class
        predicted_idx = np.argmax(prediction)
        predicted_letter = ASL_LETTERS[predicted_idx]
        assert predicted_letter in ASL_LETTERS
        
        # Confidence should be between 0 and 1
        confidence = np.max(prediction)
        assert 0 <= confidence <= 1


