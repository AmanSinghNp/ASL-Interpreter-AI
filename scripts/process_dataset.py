"""
ASL Dataset Processing Script
Converts a folder of ASL images into a CSV of landmarks for training.
Expected structure:
dataset/
  A/
    image1.jpg
    ...
  B/
    ...
"""

import cv2
import mediapipe as mp
import csv
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# Configuration
DATASET_DIR = 'dataset/asl_alphabet_train'
CSV_FILE = 'asl_data.csv'
MAX_SAMPLES_PER_CLASS = 1000 # Limit samples to keep file size manageable/balanced

# ASL letters to process (excluding J and Z)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Important for processing independent images
    max_num_hands=1,
    min_detection_confidence=0.5
)

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist and scale"""
    if not landmarks:
        return None
    
    # Convert to list of dictionaries
    landmark_list = []
    for landmark in landmarks.landmark:
        landmark_list.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    
    # Get wrist position (landmark 0)
    wrist = landmark_list[0]
    
    # Normalize relative to wrist
    normalized = []
    for lm in landmark_list:
        normalized.append({
            'x': lm['x'] - wrist['x'],
            'y': lm['y'] - wrist['y'],
            'z': lm['z'] - wrist['z']
        })
    
    # Calculate scale factor
    middle_finger_mcp = landmark_list[9]  # Middle finger MCP
    scale_x = abs(middle_finger_mcp['x'] - wrist['x'])
    scale_y = abs(middle_finger_mcp['y'] - wrist['y'])
    scale = max(scale_x, scale_y, 0.01)
    
    # Scale normalize
    scaled = []
    for lm in normalized:
        scaled.append({
            'x': lm['x'] / scale,
            'y': lm['y'] / scale,
            'z': lm['z'] / scale
        })
    
    # Extract 42 features (21 x-coords, 21 y-coords)
    features = []
    for i in range(21):
        features.append(scaled[i]['x'])
    for i in range(21):
        features.append(scaled[i]['y'])
    
    return features

def process_image(file_path):
    """Process a single image and return features"""
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            return normalize_landmarks(results.multi_hand_landmarks[0])
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return None

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please create a 'dataset' folder and place your ASL images in subfolders (A, B, C...).")
        return

    print("="*50)
    print(f"ASL Dataset Processor")
    print(f"Reading from: {DATASET_DIR}/")
    print(f"Writing to: {CSV_FILE}")
    print("="*50)

    # Initialize CSV with header
    header = ['letter'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    total_samples = 0
    
    for letter in ASL_LETTERS:
        letter_dir = os.path.join(DATASET_DIR, letter)
        if not os.path.exists(letter_dir):
            # Try lowercase
            letter_dir = os.path.join(DATASET_DIR, letter.lower())
            if not os.path.exists(letter_dir):
                print(f"Skipping {letter}: Folder not found")
                continue
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(letter_dir, ext)))
            
        print(f"Processing letter {letter}: Found {len(image_files)} images...")
        
        # Limit samples if needed
        if len(image_files) > MAX_SAMPLES_PER_CLASS:
            image_files = image_files[:MAX_SAMPLES_PER_CLASS]
        
        count = 0
        features_batch = []
        
        # Process images (sequentially to avoid threading issues with MediaPipe/OpenCV in some envs)
        # Using a simple loop is safer than ThreadPool for MediaPipe in some contexts
        for img_path in image_files:
            features = process_image(img_path)
            if features:
                features_batch.append([letter] + features)
                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count} images...", end='\r')
        
        # Write batch to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(features_batch)
            
        print(f"  ✓ Saved {count} samples for {letter}")
        total_samples += count

    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total samples collected: {total_samples}")
    print(f"Data saved to {CSV_FILE}")
    print("="*50)
    hands.close()

if __name__ == '__main__':
    main()

