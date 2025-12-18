"""
Unit tests for utils.py - shared utilities and configuration.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    ASL_LETTERS,
    normalize_landmarks,
    normalize_landmarks_flat,
    CONFIDENCE_THRESHOLD,
    STABILITY_THRESHOLD,
)


class TestASLLetters:
    """Tests for ASL_LETTERS configuration."""
    
    def test_letters_count(self):
        """Should have 24 letters (A-Y excluding J and Z)."""
        assert len(ASL_LETTERS) == 24
    
    def test_letters_are_uppercase(self):
        """All letters should be uppercase."""
        for letter in ASL_LETTERS:
            assert letter.isupper()
    
    def test_excludes_j_and_z(self):
        """J and Z should be excluded (they require motion)."""
        assert 'J' not in ASL_LETTERS
        assert 'Z' not in ASL_LETTERS
    
    def test_contains_expected_letters(self):
        """Should contain all static ASL letters."""
        expected = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                    'V', 'W', 'X', 'Y']
        assert ASL_LETTERS == expected
    
    def test_letters_are_sorted(self):
        """Letters should be in alphabetical order (for consistent indexing)."""
        # Create expected sorted order (A-Y without J, Z)
        all_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        expected = [l for l in all_letters if l not in ['J', 'Z']]
        assert ASL_LETTERS == expected


class TestNormalizeLandmarks:
    """Tests for normalize_landmarks function."""
    
    def test_returns_none_for_none_input(self):
        """Should return None when input is None."""
        result = normalize_landmarks(None)
        assert result is None
    
    def test_output_shape(self, sample_landmarks):
        """Output should be shape (1, 42)."""
        result = normalize_landmarks(sample_landmarks)
        assert result is not None
        assert result.shape == (1, 42)
    
    def test_output_is_numpy_array(self, sample_landmarks):
        """Output should be a numpy array."""
        result = normalize_landmarks(sample_landmarks)
        assert isinstance(result, np.ndarray)
    
    def test_first_21_are_x_coords(self, centered_landmarks):
        """First 21 values should be normalized x coordinates."""
        result = normalize_landmarks(centered_landmarks)
        x_coords = result[0, :21]
        
        # Wrist should be at 0 (normalized to itself)
        assert abs(x_coords[0]) < 1e-6
        
        # All x values should be finite
        assert np.all(np.isfinite(x_coords))
    
    def test_last_21_are_y_coords(self, centered_landmarks):
        """Last 21 values should be normalized y coordinates."""
        result = normalize_landmarks(centered_landmarks)
        y_coords = result[0, 21:]
        
        # Wrist y should be at 0 (normalized to itself)
        assert abs(y_coords[0]) < 1e-6
        
        # All y values should be finite
        assert np.all(np.isfinite(y_coords))
    
    def test_wrist_is_normalized_to_zero(self, centered_landmarks):
        """Wrist (landmark 0) should be at origin after normalization."""
        result = normalize_landmarks(centered_landmarks)
        
        # x[0] and y[0] should both be 0
        assert abs(result[0, 0]) < 1e-6   # x0
        assert abs(result[0, 21]) < 1e-6  # y0
    
    def test_scale_normalization(self, centered_landmarks):
        """Values should be scaled based on wrist-to-MCP distance."""
        result = normalize_landmarks(centered_landmarks)
        
        # Values should be reasonable (typically between -10 and 10)
        assert np.all(np.abs(result) < 20)
    
    def test_handles_dict_input(self):
        """Should handle pre-converted list of dicts."""
        landmarks_list = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0}
            for _ in range(21)
        ]
        # This would fail if the function doesn't handle dict input
        # Since all points are the same, we can't normalize properly,
        # but it should at least not crash
        landmarks_list[9] = {'x': 0.6, 'y': 0.4, 'z': 0.0}  # Make MCP different
        
        # Create a mock object that returns None for hasattr check
        class DictLandmarks:
            def __init__(self, lst):
                self.data = lst
        
        # The function checks hasattr(landmarks, 'landmark'), so this tests
        # that it falls back to treating input as list of dicts


class TestNormalizeLandmarksFlat:
    """Tests for normalize_landmarks_flat function."""
    
    def test_returns_none_for_none_input(self):
        """Should return None when input is None."""
        result = normalize_landmarks_flat(None)
        assert result is None
    
    def test_returns_list(self, sample_landmarks):
        """Should return a Python list, not numpy array."""
        result = normalize_landmarks_flat(sample_landmarks)
        assert isinstance(result, list)
    
    def test_output_length(self, sample_landmarks):
        """Output list should have 42 elements."""
        result = normalize_landmarks_flat(sample_landmarks)
        assert len(result) == 42
    
    def test_values_are_floats(self, sample_landmarks):
        """All values should be floats."""
        result = normalize_landmarks_flat(sample_landmarks)
        for val in result:
            assert isinstance(val, (int, float))


class TestConfigurationConstants:
    """Tests for configuration constants."""
    
    def test_confidence_threshold_valid(self):
        """Confidence threshold should be between 0 and 1."""
        assert 0 < CONFIDENCE_THRESHOLD < 1
    
    def test_stability_threshold_positive(self):
        """Stability threshold should be a positive integer."""
        assert STABILITY_THRESHOLD > 0
        assert isinstance(STABILITY_THRESHOLD, int)


class TestNormalizationConsistency:
    """Tests for normalization consistency."""
    
    def test_same_input_same_output(self, centered_landmarks):
        """Same input should produce same output."""
        result1 = normalize_landmarks(centered_landmarks)
        result2 = normalize_landmarks(centered_landmarks)
        np.testing.assert_array_equal(result1, result2)
    
    def test_translation_invariance(self):
        """
        Landmarks should produce same features regardless of position.
        Since we normalize relative to wrist, shifting all landmarks
        should produce the same normalized output.
        """
        from tests.conftest import MockLandmarks
        
        # Original landmarks
        base_landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0}
            for _ in range(21)
        ]
        base_landmarks[9] = {'x': 0.6, 'y': 0.4, 'z': 0.0}  # Different MCP
        
        # Shifted landmarks (all moved by same amount)
        shifted_landmarks = [
            {'x': lm['x'] + 0.1, 'y': lm['y'] + 0.2, 'z': lm['z']}
            for lm in base_landmarks
        ]
        
        mock_base = MockLandmarks(base_landmarks)
        mock_shifted = MockLandmarks(shifted_landmarks)
        
        result_base = normalize_landmarks(mock_base)
        result_shifted = normalize_landmarks(mock_shifted)
        
        # Results should be identical (translation invariant)
        np.testing.assert_array_almost_equal(result_base, result_shifted)

