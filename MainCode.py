import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
DATASET_PATH = "dataset"
gestures = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]
data, labels = [], []

for label, gesture in enumerate(gestures):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    for file in os.listdir(gesture_path):
        if file.endswith(".npy"):
            try:
                gesture_data = np.load(os.path.join(gesture_path, file))
                data.append(gesture_data)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")

# Ensure data is not empty
if len(data) == 0:
    print("Error: No gesture data found! Make sure the dataset folder has .npy files.")
    exit()

data, labels = np.array(data), np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model with scaling
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42))
model.fit(X_train, y_train)
print(f"Model trained. Accuracy: {model.score(X_test, y_test):.2f}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

sentence = []
last_detected_gesture = None
last_detection_time = time.time()
min_confidence_threshold = 0.5  # Adjusted for better detection

def is_valid_gesture(prediction, probabilities, threshold=min_confidence_threshold):
    return probabilities[prediction] >= threshold

print("Press 'q' to exit. Press 's' to display the sentence.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    detected_gesture = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            # Check shape compatibility
            if landmarks.shape[1] != data.shape[1]:
                print(f"Error: Expected input shape {data.shape[1]}, but got {landmarks.shape[1]}")
                continue

            # Predict gesture
            probabilities = model.predict_proba(landmarks)[0]
            prediction = np.argmax(probabilities)

            if is_valid_gesture(prediction, probabilities) and prediction != last_detected_gesture:
                detected_gesture = gestures[prediction]
                last_detected_gesture = prediction
                last_detection_time = time.time()
                print(f"Detected: {detected_gesture} (Confidence: {probabilities[prediction]:.2f})")

    # Add detected word if enough time has passed
    if detected_gesture and (not sentence or detected_gesture != sentence[-1]):
        if time.time() - last_detection_time > 0.5:  # Prevent duplicates
            sentence.append(detected_gesture)

    # Ensure detected sentence appears on the webcam (Red color)
    cv2.putText(frame, "Current Sentence:", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, " ".join(sentence), (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        complete_sentence = " ".join(sentence)
        print("Complete Sentence:", complete_sentence)
        sentence = []  # Reset for new sentence
        last_detected_gesture = None

cap.release()
cv2.destroyAllWindows()
