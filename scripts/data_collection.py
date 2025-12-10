"""
ASL Data Collection Script
Collects hand landmark data using MediaPipe and OpenCV
Press letter keys (A-Z, excluding J and Z) to record hand landmarks
Press 'q' to quit
"""

import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ASL letters (excluding J and Z as they require motion)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# CSV file to store data
CSV_FILE = 'asl_data.csv'
samples_per_letter = 100

def normalize_landmarks(landmarks, image_width, image_height):
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

def save_landmarks_to_csv(letter, features):
    """Save normalized landmarks to CSV file"""
    file_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists:
            header = ['letter'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            writer.writerow(header)
        
        # Write data row
        row = [letter] + features
        writer.writerow(row)

def main():
    cap = cv2.VideoCapture(0)
    current_letter = None
    sample_count = {}
    
    # Initialize sample counts
    for letter in ASL_LETTERS:
        sample_count[letter] = 0
    
    print("\n" + "="*50)
    print("ASL Data Collection Tool")
    print("="*50)
    print(f"\nTarget: {samples_per_letter} samples per letter")
    print(f"Letters: {', '.join(ASL_LETTERS)}")
    print("\nInstructions:")
    print("- Press a letter key (A-Y, excluding J/Z) to start recording")
    print("- Show that ASL letter sign to the camera")
    print("- The script will record landmarks automatically")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset current letter")
    print("="*50 + "\n")
    
    recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        # Display status
        status_text = "Press letter key to start recording"
        if current_letter:
            count = sample_count.get(current_letter, 0)
            status_text = f"Recording: {current_letter} ({count}/{samples_per_letter})"
            
            # Auto-record when hand is detected
            if results.multi_hand_landmarks and recording:
                features = normalize_landmarks(
                    results.multi_hand_landmarks[0],
                    frame.shape[1],
                    frame.shape[0]
                )
                
                if features:
                    save_landmarks_to_csv(current_letter, features)
                    sample_count[current_letter] = count + 1
                    recording = False  # Wait before next recording
        
        # Draw text on frame
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show sample counts
        y_offset = 60
        for letter in ASL_LETTERS[:6]:  # Show first 6 letters
            count = sample_count[letter]
            text = f"{letter}: {count}/{samples_per_letter}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow('ASL Data Collection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_letter = None
            recording = False
            print("Reset - select a letter to start recording")
        elif key >= ord('a') and key <= ord('z'):
            letter = chr(key).upper()
            if letter in ASL_LETTERS:
                current_letter = letter
                count = sample_count.get(letter, 0)
                if count < samples_per_letter:
                    recording = True
                    print(f"\nRecording samples for letter: {letter}")
                    print(f"Current count: {count}/{samples_per_letter}")
                else:
                    print(f"\nLetter {letter} already has {samples_per_letter} samples!")
            elif letter in ['J', 'Z']:
                print(f"\nLetter {letter} is excluded (requires motion)")
        
        # Auto-advance when target reached
        if current_letter and sample_count.get(current_letter, 0) >= samples_per_letter:
            print(f"\n✓ Completed {samples_per_letter} samples for {current_letter}!")
            current_letter = None
            recording = False
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n" + "="*50)
    print("Data Collection Complete!")
    print(f"Data saved to: {CSV_FILE}")
    print("\nSample counts:")
    for letter in ASL_LETTERS:
        count = sample_count.get(letter, 0)
        print(f"  {letter}: {count} samples")
    print("="*50)

if __name__ == '__main__':
    main()


