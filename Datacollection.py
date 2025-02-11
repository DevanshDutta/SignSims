import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create dataset folder if not exists
DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

def get_gesture_name():
    """Function to get a new gesture name from the user."""
    gesture_name = input("\nEnter the name of the gesture: ").strip()
    gesture_path = os.path.join(DATASET_PATH, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)
    return gesture_name, gesture_path

# Get the first gesture name
gesture_name, gesture_path = get_gesture_name()

# Start capturing
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sample_num = 0
print("Press 'S' to capture a sample.")
print("Press 'W' to change the gesture name.")
print("Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF

            # Save landmarks when 'S' is pressed
            if key == ord('s'):
                file_path = os.path.join(gesture_path, f"{sample_num}.npy")
                np.save(file_path, np.array(landmarks))
                sample_num += 1
                print(f"Captured sample {sample_num} for gesture: {gesture_name}")

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'W' to enter a new gesture name
    if key == ord('w'):
        gesture_name, gesture_path = get_gesture_name()
        sample_num = 0  # Reset sample number for new gesture

    # Press 'Q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
