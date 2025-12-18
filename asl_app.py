"""
ASL Interpreter - Desktop Application
Real-time American Sign Language recognition using MediaPipe and TensorFlow.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

from utils import MODEL_PATH, CONFIDENCE_THRESHOLD, STABILITY_THRESHOLD, normalize_landmarks, load_classes

# --- Model Loading ---
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # Load the specific classes this model was trained on (saved during training)
    ASL_LETTERS = load_classes()
    out_dim = model.output_shape[-1]
    if out_dim != len(ASL_LETTERS):
        print("Error: Model output size does not match loaded class labels.")
        print(f"  model.output_shape[-1] = {out_dim}")
        print(f"  len(classes)          = {len(ASL_LETTERS)}")
        print("Fix: re-run processing + training so saved_model/classes.txt matches the model.")
        exit(1)

    print(f"Loaded {len(ASL_LETTERS)} classes: {', '.join(ASL_LETTERS)}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    print("Run `python -m scripts.train_model` first if you haven't.")
    exit(1)

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def draw_confidence_bar(frame, confidence, x, y, width=200, height=15):
    """Draw a confidence meter bar on the frame."""
    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    # Filled portion based on confidence
    fill_width = int(confidence * width)
    # Color gradient: red -> yellow -> green
    if confidence < 0.5:
        color = (0, 0, 255)  # Red
    elif confidence < CONFIDENCE_THRESHOLD:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 255, 0)  # Green
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)


def draw_ui_overlay(frame, fps, prediction_text, confidence, current_word, flash_letter=None, flash_alpha=0):
    """
    Draw the user interface overlay on the frame.
    
    Args:
        frame: OpenCV frame to draw on
        fps: Current frames per second
        prediction_text: Text showing current prediction
        confidence: Confidence value (0-1)
        current_word: Currently constructed word
        flash_letter: Letter to flash (when confirmed)
        flash_alpha: Alpha value for flash effect (0-1)
    """
    h, w, _ = frame.shape
    
    # --- Top Panel ---
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    
    # Prediction text
    cv2.putText(frame, f"Sign: {prediction_text}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
    
    # Confidence bar
    cv2.putText(frame, "Confidence:", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    draw_confidence_bar(frame, confidence, 120, 60, 150, 20)
    
    # FPS counter (top right)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # --- Bottom Panel ---
    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    
    # Word display with cursor
    word_display = current_word + "|"
    cv2.putText(frame, "Word:", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, word_display, (90, h - 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "[Space] Add space  [Backspace] Delete  [C] Clear  [Q] Quit",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    # --- Letter Confirmation Flash ---
    if flash_letter and flash_alpha > 0:
        # Create semi-transparent overlay for flash effect
        overlay = frame.copy()
        
        # Large letter in center
        text_size = cv2.getTextSize(flash_letter, cv2.FONT_HERSHEY_DUPLEX, 4, 4)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Draw letter with glow effect
        glow_color = (0, int(255 * flash_alpha), 0)
        cv2.putText(overlay, flash_letter, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 4, glow_color, 6)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, flash_alpha * 0.3, frame, 1 - flash_alpha * 0.3, 0, frame)


def main():
    """Main application loop."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Make sure your webcam is connected and not in use by another application.")
        exit(1)
    
    # Application state
    current_word = ""
    prediction_buffer = []
    last_stable_letter = None
    
    # Flash animation state
    flash_letter = None
    flash_start_time = 0
    flash_duration = 0.4  # seconds
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    print("\n" + "=" * 50)
    print("ASL Interpreter - Desktop App")
    print("=" * 50)
    print("Controls:")
    print(" - Show hand to sign")
    print(" - 'Space' key: Add space")
    print(" - 'Backspace' key: Delete last character")
    print(" - 'C' key: Clear word")
    print(" - 'Q' key: Quit")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time
        
        # Process hand detection
        results = hands.process(rgb_frame)
        
        prediction_text = "No hand detected"
        confidence_val = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with custom styling
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract and normalize features
                features = normalize_landmarks(hand_landmarks)
                
                if features is not None:
                    try:
                        # Get model prediction
                        prediction = model.predict(features, verbose=0)
                        predicted_index = np.argmax(prediction)
                        confidence_val = float(np.max(prediction))
                        predicted_letter = ASL_LETTERS[predicted_index]
                        
                        if confidence_val > CONFIDENCE_THRESHOLD:
                            # Add to prediction buffer for stability
                            prediction_buffer.append(predicted_letter)
                            if len(prediction_buffer) > STABILITY_THRESHOLD:
                                prediction_buffer.pop(0)
                            
                            # Check for stable prediction
                            if (len(prediction_buffer) == STABILITY_THRESHOLD and
                                    all(p == predicted_letter for p in prediction_buffer)):
                                
                                prediction_text = f"{predicted_letter} ({confidence_val:.0%})"
                                
                                # Add letter if it's different from last stable letter
                                if predicted_letter != last_stable_letter:
                                    current_word += predicted_letter
                                    last_stable_letter = predicted_letter
                                    
                                    # Trigger flash animation
                                    flash_letter = predicted_letter
                                    flash_start_time = current_time
                                    
                                    # Reset buffer to require re-stabilization
                                    prediction_buffer = []
                            else:
                                prediction_text = f"{predicted_letter} ({confidence_val:.0%}) ..."
                        else:
                            prediction_text = f"Low confidence ({confidence_val:.0%})"
                            prediction_buffer = []
                            last_stable_letter = None
                            
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        prediction_text = "Prediction error"
        else:
            # No hand detected - reset state
            prediction_buffer = []
            last_stable_letter = None
        
        # Calculate flash animation alpha
        flash_alpha = 0
        if flash_letter:
            elapsed = current_time - flash_start_time
            if elapsed < flash_duration:
                # Fade out effect
                flash_alpha = 1.0 - (elapsed / flash_duration)
            else:
                flash_letter = None
        
        # Draw UI overlay
        draw_ui_overlay(frame, fps, prediction_text, confidence_val, 
                        current_word, flash_letter, flash_alpha)
        
        # Display frame
        cv2.imshow('ASL Interpreter', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32:  # Space
            current_word += " "
            last_stable_letter = None  # Allow same letter after space
        elif key == 8:  # Backspace
            current_word = current_word[:-1]
        elif key == ord('c') or key == ord('C'):
            current_word = ""
            last_stable_letter = None
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\nSession ended.")
    if current_word:
        print(f"Final word: {current_word}")


if __name__ == '__main__':
    main()
