# SignSims

SignSims is an advanced sign language detection system that goes beyond recognizing individual letters, allowing the detection of complete words and the formation of sentences in real-time. Built using MediaPipe, OpenCV, and machine learning algorithms, this project is designed to bridge communication gaps for sign language users by recognizing hand gestures and translating them into text.

The project includes two core components:
1. Data Collection: A custom data collection tool that captures and stores hand landmarks from webcam input. Users can define and collect gesture data for each word in sign language, creating a robust dataset for training.
2. Sign Language Detection: A real-time detection system that processes webcam feed, detects hand landmarks, and uses a machine learning model to classify hand gestures into words rather than just individual letters. As gestures are detected, the system can construct full sentences from recognized words.
