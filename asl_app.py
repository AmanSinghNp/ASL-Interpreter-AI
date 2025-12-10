import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# --- Configuration ---
MODEL_PATH = 'saved_model/asl_model'
CONFIDENCE_THRESHOLD = 0.7
STABILITY_THRESHOLD = 5  # Number of consistent frames for stability
# Letters A-Z excluding J and Z (must match training order)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# --- Model Loading ---
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    print("Run scripts/train_model.py first if you haven't.")
    exit(1)

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper Functions ---
def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist and scale
    Matches the logic in scripts/data_collection.py exactly
    """
    if not landmarks:
        return None
    
    # Convert to list of dictionaries for easier processing
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
    
    # Calculate scale factor (distance from wrist to middle finger MCP)
    middle_finger_mcp = landmark_list[9]
    scale_x = abs(middle_finger_mcp['x'] - wrist['x'])
    scale_y = abs(middle_finger_mcp['y'] - wrist['y'])
    scale = max(scale_x, scale_y, 0.01)
    
    # Scale normalize and flatten
    features = []
    for lm in normalized:
        features.append(lm['x'] / scale)
    for lm in normalized:
        features.append(lm['y'] / scale)
    
    return np.array([features])

# --- Main App Logic ---
def main():
    cap = cv2.VideoCapture(0)
    
    # UI State
    current_word = ""
    prediction_buffer = []
    
    print("\n" + "="*50)
    print("ASL Interpreter - Desktop App")
    print("="*50)
    print("Controls:")
    print(" - Show hand to sign")
    print(" - 'Space' key: Add space")
    print(" - 'Backsapce' key: Delete last character")
    print(" - 'c' key: Clear word")
    print(" - 'q' key: Quit")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process Hand
        results = hands.process(rgb_frame)
        prediction_text = "..."
        confidence_val = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract features
                features = normalize_landmarks(hand_landmarks)
                
                # Predict
                if features is not None:
                    try:
                        prediction = model.predict(features, verbose=0)
                        predicted_index = np.argmax(prediction)
                        confidence_val = np.max(prediction)
                        
                        predicted_letter = ASL_LETTERS[predicted_index]
                        
                        if confidence_val > CONFIDENCE_THRESHOLD:
                            prediction_buffer.append(predicted_letter)
                            if len(prediction_buffer) > STABILITY_THRESHOLD:
                                prediction_buffer.pop(0)
                            
                            # Check stability (all recent predictions match)
                            if len(prediction_buffer) == STABILITY_THRESHOLD and \
                               all(p == predicted_letter for p in prediction_buffer):
                                
                                prediction_text = f"{predicted_letter} ({confidence_val:.2f})"
                        else:
                            prediction_text = "Low Confidence"
                            prediction_buffer = []
                            
                    except Exception as e:
                        print(f"Prediction error: {e}")
        else:
            prediction_buffer = []

        # Draw UI
        # Background for text
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Current Prediction
        cv2.putText(frame, f"Sign: {prediction_text}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        # Current Word
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Word: {current_word}", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('ASL Interpreter', frame)
        
        # Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Space
            current_word += " "
        elif key == 8: # Backspace
            current_word = current_word[:-1]
        elif key == ord('c'):
            current_word = ""

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()

