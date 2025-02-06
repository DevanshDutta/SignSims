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
            data.append(np.load(os.path.join(gesture_path, file)))
            labels.append(label)

data, labels = np.array(data), np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model with scaling for better accuracy
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(X_train, y_train)
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start detection
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

sentence = []
sentence_start_time = time.time()
SENTENCE_TIME_LIMIT = 5  # 5 seconds to complete the sentence
DISPLAY_TIME = 5  # 5 seconds to display the sentence
last_detected_gesture = None

def is_valid_gesture(prediction, probabilities, threshold=0.8):
    return probabilities[prediction] >= threshold

print("Press 'q' to exit. Press 's' to stop detection and display the sentence.")

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

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            if landmarks.shape[1] == data.shape[1]:  # Ensure consistent dimensions
                probabilities = model.predict_proba(landmarks)[0]
                prediction = np.argmax(probabilities)
                
                if is_valid_gesture(prediction, probabilities) and prediction != last_detected_gesture:
                    detected_gesture = gestures[prediction]
                    last_detected_gesture = prediction

    if detected_gesture and (not sentence or detected_gesture != sentence[-1]):  # Avoid consecutive duplicates
        sentence.append(detected_gesture)
    
    cv2.putText(frame, f"Current Sentence: {' '.join(sentence)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        complete_sentence = " ".join(sentence)
        print("Complete Sentence:", complete_sentence)
        start_display_time = time.time()
        while time.time() - start_display_time < DISPLAY_TIME:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, complete_sentence, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.imshow("Sign Language Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        sentence = []  # Reset for new sentence
        sentence_start_time = time.time()
        last_detected_gesture = None

cap.release()
cv2.destroyAllWindows()