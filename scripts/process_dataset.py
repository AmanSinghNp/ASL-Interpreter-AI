"""
ASL Dataset Processing Script
Converts a folder of ASL images into a CSV of landmarks for training.

Expected folder structure:
    dataset/asl_alphabet_train/
        A/
            image1.jpg
            image2.jpg
            ...
        B/
            ...

Usage:
    python -m scripts.process_dataset
    python -m scripts.process_dataset --dataset_dir path/to/dataset --max_samples 500
"""

import sys
import os
import argparse
import glob

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import csv

from utils import ASL_LETTERS, CSV_FILE, DATASET_DIR, MAX_SAMPLES_PER_CLASS, normalize_landmarks_flat


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process ASL image dataset into CSV')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                        help=f'Path to dataset directory (default: {DATASET_DIR})')
    parser.add_argument('--output', type=str, default=CSV_FILE,
                        help=f'Output CSV file path (default: {CSV_FILE})')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES_PER_CLASS,
                        help=f'Max samples per class (default: {MAX_SAMPLES_PER_CLASS})')
    return parser.parse_args()


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Important for processing independent images
    max_num_hands=1,
    min_detection_confidence=0.5
)


def process_image(file_path):
    """Process a single image and return normalized features."""
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            return normalize_landmarks_flat(results.multi_hand_landmarks[0])
            
    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {e}")
    
    return None


def main():
    """Main processing function."""
    args = parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' not found.")
        print("\nExpected folder structure:")
        print(f"  {args.dataset_dir}/")
        print("    A/")
        print("      image1.jpg")
        print("      ...")
        print("    B/")
        print("      ...")
        return

    print("=" * 60)
    print("ASL Dataset Processor")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Dataset directory: {args.dataset_dir}")
    print(f"  - Output file: {args.output}")
    print(f"  - Max samples per class: {args.max_samples}")
    print(f"  - Letters to process: {', '.join(ASL_LETTERS)}")

    # Initialize CSV with header
    header = ['letter'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    total_samples = 0
    letter_stats = {}
    
    print("\n" + "-" * 60)
    print("Processing images...")
    print("-" * 60)
    
    for letter in ASL_LETTERS:
        # Try to find letter directory (uppercase or lowercase)
        letter_dir = os.path.join(args.dataset_dir, letter)
        if not os.path.exists(letter_dir):
            letter_dir = os.path.join(args.dataset_dir, letter.lower())
            if not os.path.exists(letter_dir):
                print(f"\n[SKIP] {letter}: Folder not found")
                letter_stats[letter] = {'found': 0, 'processed': 0}
                continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(letter_dir, ext)))
        
        if not image_files:
            print(f"\n[SKIP] {letter}: No images found")
            letter_stats[letter] = {'found': 0, 'processed': 0}
            continue
            
        print(f"\n[{letter}] Found {len(image_files)} images...")
        
        # Limit samples if needed
        if len(image_files) > args.max_samples:
            image_files = image_files[:args.max_samples]
            print(f"     Limited to {args.max_samples} samples")
        
        # Process images
        count = 0
        features_batch = []
        
        for i, img_path in enumerate(image_files):
            features = process_image(img_path)
            if features:
                features_batch.append([letter] + features)
                count += 1
            
            # Progress update every 100 images
            if (i + 1) % 100 == 0:
                print(f"     Progress: {i + 1}/{len(image_files)} images processed...", end='\r')
        
        # Write batch to CSV
        if features_batch:
            with open(args.output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(features_batch)
        
        success_rate = (count / len(image_files)) * 100 if image_files else 0
        print(f"     ✓ Saved {count}/{len(image_files)} samples ({success_rate:.1f}% success)")
        
        letter_stats[letter] = {'found': len(image_files), 'processed': count}
        total_samples += count

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Output file: {args.output}")
    print(f"  - Total samples: {total_samples}")
    
    print(f"\nPer-letter breakdown:")
    for letter in ASL_LETTERS:
        stats = letter_stats.get(letter, {'found': 0, 'processed': 0})
        if stats['found'] > 0:
            print(f"  {letter}: {stats['processed']:4d} / {stats['found']:4d} images")
        else:
            print(f"  {letter}:    - / -    (no data)")
    
    print("\n" + "=" * 60)
    print(f"Next step: Run 'python -m scripts.train_model' to train the model")
    print("=" * 60)
    
    hands.close()


if __name__ == '__main__':
    main()
