"""
ASL Data Collection Script
Collects hand landmark data using MediaPipe and OpenCV for training.

Usage:
    python -m scripts.data_collection

Instructions:
    - Press letter keys (A-Y, excluding J/Z) to start recording
    - Show that ASL letter sign to the camera
    - The script automatically records samples
    - Press 'Q' to quit
    - Press 'R' to reset current letter
"""

import sys
import os

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import csv

from utils import ASL_LETTERS, CSV_FILE, normalize_landmarks_flat

# Configuration
SAMPLES_PER_LETTER = 100

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def save_landmarks_to_csv(letter, features, csv_file):
    """Save normalized landmarks to CSV file."""
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists:
            header = ['letter'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            writer.writerow(header)
        
        # Write data row
        row = [letter] + features
        writer.writerow(row)


def draw_ui(frame, current_letter, sample_count, recording):
    """Draw the user interface overlay."""
    h, w, _ = frame.shape
    
    # Status bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    
    if current_letter:
        count = sample_count.get(current_letter, 0)
        progress = count / SAMPLES_PER_LETTER
        
        # Status text
        status = f"Recording: {current_letter} ({count}/{SAMPLES_PER_LETTER})"
        color = (0, 255, 0) if recording else (0, 165, 255)
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Progress bar
        bar_width = int(progress * 200)
        cv2.rectangle(frame, (20, 45), (220, 55), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 45), (20 + bar_width, 55), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 45), (220, 55), (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Press a letter key (A-Y) to start recording",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Sample count panel on right side
    panel_x = w - 180
    cv2.rectangle(frame, (panel_x, 0), (w, min(h, 400)), (20, 20, 20), -1)
    cv2.putText(frame, "Sample Counts:", (panel_x + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset = 50
    for i, letter in enumerate(ASL_LETTERS):
        count = sample_count.get(letter, 0)
        
        # Highlight current letter
        if letter == current_letter:
            cv2.rectangle(frame, (panel_x + 5, y_offset - 12), (w - 5, y_offset + 3), (50, 50, 50), -1)
        
        # Color based on completion
        if count >= SAMPLES_PER_LETTER:
            color = (0, 255, 0)  # Green - complete
        elif count > 0:
            color = (0, 165, 255)  # Orange - in progress
        else:
            color = (100, 100, 100)  # Gray - not started
        
        text = f"{letter}: {count:3d}"
        cv2.putText(frame, text, (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 15
        
        # Start second column after 12 letters
        if i == 11:
            y_offset = 50
            panel_x = w - 90
    
    # Instructions at bottom
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[Q] Quit  [R] Reset letter  [Letter key] Select letter",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)


def main():
    """Main data collection loop."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    current_letter = None
    sample_count = {letter: 0 for letter in ASL_LETTERS}
    recording = False
    frame_delay = 0  # Frames to wait between recordings
    
    print("\n" + "=" * 50)
    print("ASL Data Collection Tool")
    print("=" * 50)
    print(f"\nTarget: {SAMPLES_PER_LETTER} samples per letter")
    print(f"Letters: {', '.join(ASL_LETTERS)}")
    print(f"Output: {CSV_FILE}")
    print("\nInstructions:")
    print("  - Press a letter key (A-Y, excluding J/Z) to start recording")
    print("  - Show that ASL letter sign to the camera")
    print("  - Hold the sign steady - samples record automatically")
    print("  - Press 'Q' to quit")
    print("  - Press 'R' to reset/deselect current letter")
    print("=" * 50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Decrement frame delay
        if frame_delay > 0:
            frame_delay -= 1
        
        # Auto-record when hand is detected and letter is selected
        if current_letter and results.multi_hand_landmarks and frame_delay == 0:
            count = sample_count.get(current_letter, 0)
            
            if count < SAMPLES_PER_LETTER:
                features = normalize_landmarks_flat(results.multi_hand_landmarks[0])
                
                if features:
                    save_landmarks_to_csv(current_letter, features, CSV_FILE)
                    sample_count[current_letter] = count + 1
                    recording = True
                    frame_delay = 3  # Wait 3 frames before next recording
                    
                    # Print progress
                    if (count + 1) % 10 == 0:
                        print(f"  {current_letter}: {count + 1}/{SAMPLES_PER_LETTER} samples")
            else:
                recording = False
        else:
            recording = False
        
        # Draw UI
        draw_ui(frame, current_letter, sample_count, recording)
        
        cv2.imshow('ASL Data Collection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            current_letter = None
            print("\nReset - select a letter to start recording")
        elif key >= ord('a') and key <= ord('z'):
            letter = chr(key).upper()
            if letter in ASL_LETTERS:
                current_letter = letter
                count = sample_count.get(letter, 0)
                if count < SAMPLES_PER_LETTER:
                    print(f"\nRecording samples for: {letter} ({count}/{SAMPLES_PER_LETTER})")
                else:
                    print(f"\nLetter {letter} complete! ({SAMPLES_PER_LETTER} samples)")
            elif letter in ['J', 'Z']:
                print(f"\nNote: Letter {letter} is excluded (requires motion)")
        
        # Auto-advance when target reached
        if current_letter and sample_count.get(current_letter, 0) >= SAMPLES_PER_LETTER:
            print(f"\n✓ Completed {SAMPLES_PER_LETTER} samples for {current_letter}!")
            current_letter = None
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Data Collection Complete!")
    print(f"Data saved to: {CSV_FILE}")
    print("\nSample counts:")
    total = 0
    for letter in ASL_LETTERS:
        count = sample_count.get(letter, 0)
        status = "✓" if count >= SAMPLES_PER_LETTER else " "
        print(f"  {status} {letter}: {count} samples")
        total += count
    print(f"\nTotal samples: {total}")
    print("=" * 50)


if __name__ == '__main__':
    main()
