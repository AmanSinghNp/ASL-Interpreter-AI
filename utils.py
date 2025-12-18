"""
ASL Interpreter AI - Shared Utilities
Centralized configuration and helper functions used across all scripts.
"""

import numpy as np

# --- Configuration Constants ---
MODEL_PATH = 'saved_model/asl_model'
CONFIDENCE_THRESHOLD = 0.7
STABILITY_THRESHOLD = 5  # Number of consistent frames for stability
CSV_FILE = 'asl_data.csv'
DATASET_DIR = 'dataset/asl_alphabet_train'
MAX_SAMPLES_PER_CLASS = 1000

# ASL letters (A-Z excluding J and Z as they require motion)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks relative to wrist position and scale.
    
    This function takes MediaPipe hand landmarks and converts them to a
    normalized feature vector suitable for model input.
    
    Args:
        landmarks: MediaPipe hand landmarks object with .landmark attribute,
                   or None if no hand detected.
    
    Returns:
        numpy.ndarray: Shape (1, 42) array of normalized features
                       (21 x-coords followed by 21 y-coords), or None if
                       no landmarks provided.
    
    The normalization process:
    1. Translates all points relative to wrist (landmark 0)
    2. Scales based on distance from wrist to middle finger MCP (landmark 9)
    3. Extracts only x and y coordinates (42 features total)
    """
    if landmarks is None:
        return None
    
    # Handle both MediaPipe landmark objects and pre-converted lists
    if hasattr(landmarks, 'landmark'):
        landmark_list = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z} 
            for lm in landmarks.landmark
        ]
    else:
        landmark_list = landmarks
    
    if not landmark_list:
        return None
    
    # Get wrist position (landmark 0) as reference point
    wrist = landmark_list[0]
    
    # Normalize all landmarks relative to wrist position
    normalized = [
        {
            'x': lm['x'] - wrist['x'],
            'y': lm['y'] - wrist['y'],
            'z': lm['z'] - wrist['z']
        }
        for lm in landmark_list
    ]
    
    # Calculate scale factor using distance from wrist to middle finger MCP
    middle_finger_mcp = landmark_list[9]
    scale_x = abs(middle_finger_mcp['x'] - wrist['x'])
    scale_y = abs(middle_finger_mcp['y'] - wrist['y'])
    scale = max(scale_x, scale_y, 0.01)  # Prevent division by zero
    
    # Scale normalize and flatten to feature vector
    # Format: [x0, x1, ..., x20, y0, y1, ..., y20]
    features = []
    for lm in normalized:
        features.append(lm['x'] / scale)
    for lm in normalized:
        features.append(lm['y'] / scale)
    
    return np.array([features])


def normalize_landmarks_flat(landmarks):
    """
    Normalize landmarks and return as flat list (for CSV storage).
    
    Same as normalize_landmarks() but returns a flat Python list
    instead of a numpy array. Used for data collection/processing.
    
    Args:
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Flat list of 42 normalized features, or None if no landmarks.
    """
    result = normalize_landmarks(landmarks)
    if result is not None:
        return result[0].tolist()
    return None

