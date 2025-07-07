import cv2
import numpy as np
from keras.src.saving.saving_api import load_model

# Load the pre-trained model
model = load_model('MoodDetection.keras')

# Define the labels for the classes (modify based on your training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the input frame for the model
def preprocess_frame(frame, target_size=(48, 48)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, target_size)  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)  # Add channel dimension
    batch_frame = np.expand_dims(expanded_frame, axis=0)  # Add batch dimension
    return batch_frame

# Start video capture
cap = cv2.VideoCapture(0) 

print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Define a region of interest (ROI) for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face for the model
        preprocessed_face = preprocess_frame(face_roi)

        # Predict emotion
        predictions = model.predict(preprocessed_face)
        emotion = emotion_labels[np.argmax(predictions)]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with the detected face and emotion
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
