"""
Pytest configuration and shared fixtures for ASL Interpreter tests.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockLandmark:
    """Mock MediaPipe landmark point."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarks:
    """Mock MediaPipe hand landmarks object."""
    def __init__(self, landmarks_list):
        self.landmark = [
            MockLandmark(lm['x'], lm['y'], lm['z'])
            for lm in landmarks_list
        ]


@pytest.fixture
def sample_landmarks():
    """
    Create sample hand landmarks for testing.
    Returns a MockLandmarks object with 21 landmarks.
    """
    # Create a simple hand shape with wrist at origin
    landmarks = []
    for i in range(21):
        # Create landmarks with some variation
        landmarks.append({
            'x': 0.5 + (i % 5) * 0.05,
            'y': 0.5 + (i // 5) * 0.1,
            'z': 0.0
        })
    
    return MockLandmarks(landmarks)


@pytest.fixture
def centered_landmarks():
    """
    Create hand landmarks centered at (0.5, 0.5).
    This represents a typical detected hand position.
    """
    # Wrist at center
    landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0}]
    
    # Thumb (landmarks 1-4)
    landmarks.extend([
        {'x': 0.45, 'y': 0.45, 'z': 0.0},
        {'x': 0.40, 'y': 0.40, 'z': 0.0},
        {'x': 0.35, 'y': 0.38, 'z': 0.0},
        {'x': 0.30, 'y': 0.36, 'z': 0.0},
    ])
    
    # Index finger (landmarks 5-8)
    landmarks.extend([
        {'x': 0.48, 'y': 0.35, 'z': 0.0},
        {'x': 0.47, 'y': 0.25, 'z': 0.0},
        {'x': 0.46, 'y': 0.18, 'z': 0.0},
        {'x': 0.45, 'y': 0.12, 'z': 0.0},
    ])
    
    # Middle finger (landmarks 9-12)
    landmarks.extend([
        {'x': 0.50, 'y': 0.33, 'z': 0.0},  # MCP - used for scale
        {'x': 0.50, 'y': 0.22, 'z': 0.0},
        {'x': 0.50, 'y': 0.14, 'z': 0.0},
        {'x': 0.50, 'y': 0.08, 'z': 0.0},
    ])
    
    # Ring finger (landmarks 13-16)
    landmarks.extend([
        {'x': 0.53, 'y': 0.35, 'z': 0.0},
        {'x': 0.54, 'y': 0.25, 'z': 0.0},
        {'x': 0.55, 'y': 0.18, 'z': 0.0},
        {'x': 0.56, 'y': 0.13, 'z': 0.0},
    ])
    
    # Pinky (landmarks 17-20)
    landmarks.extend([
        {'x': 0.56, 'y': 0.38, 'z': 0.0},
        {'x': 0.58, 'y': 0.30, 'z': 0.0},
        {'x': 0.60, 'y': 0.24, 'z': 0.0},
        {'x': 0.62, 'y': 0.20, 'z': 0.0},
    ])
    
    return MockLandmarks(landmarks)


