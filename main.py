import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import imutils
import time
import dlib
import playsound
import streamlit as st
import requests

# ---------------- Streamlit UI Setup ----------------
st.set_page_config(layout="wide")
st.title("🚗 Smart Driver & Vehicle Monitoring Dashboard")

col1, col2 = st.columns(2)

status_placeholder = col1.empty()
frame_placeholder = col1.empty()

car_status_placeholder = col2.empty()

# ---------------- Load Models ----------------
road_rage_model = load_model('Models/road_rage_detection_model2.h5')
mood_model = load_model('Models/MoodDetection.keras')
drowsiness_predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    st.error("❌ Error: Haar cascade not loaded. Check path.")
    st.stop()

# ---------------- Constants ----------------
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------- Helper Functions ----------------
def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        playsound.playsound(path)
    if alarm_status2:
        saying = True
        playsound.playsound(path)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

def preprocess_frame(frame, target_size=(48, 48)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, target_size)
    normalized_frame = resized_frame / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    return np.expand_dims(expanded_frame, axis=0)

# ---------------- Main Function ----------------
def main():
    global alarm_status, alarm_status2, COUNTER
    frame_buffer = []

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- Drowsiness Detection -----
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = drowsiness_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=sound_alarm, args=("Alert.wav",), daemon=True).start()
                    status_placeholder.markdown("### 🛑 DROWSINESS ALERT!")
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                status_placeholder.markdown("### 🛑 Yawn Alert!")
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    Thread(target=sound_alarm, args=("Alert.wav",), daemon=True).start()
            else:
                alarm_status2 = False

        # ----- Mood Detection -----
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_frame(face_roi)
            predictions = mood_model.predict(preprocessed_face)
            emotion = emotion_labels[np.argmax(predictions)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            status_placeholder.markdown(f"### 😊 Mood Detected: {emotion}")

        # ----- Road Rage Detection -----
        frame_resized = cv2.resize(frame, (150, 150))
        frame_buffer.append(frame_resized)

        if len(frame_buffer) == 30:
            input_data = np.array(frame_buffer).reshape((1, 30, 150, 150, 3)) / 255.0
            prediction = road_rage_model.predict(input_data)[0][0]
            label = "Violence" if prediction > 0.5 else "No Violence"
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            status_placeholder.markdown(f"### 🚨 Road Rage: {label}")
            frame_buffer.pop(0)

        # Show video in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # ----- Vehicle Status API -----
        try:
            res = requests.get("http://localhost:5000/get")
            if res.ok:
                data = res.json()
                if "message" in data:
                    car_status_placeholder.markdown(f"### 🛑 {data['message']}")
                else:
                    car_status_placeholder.markdown("### ✅ Vehicle Running")
            else:
                car_status_placeholder.markdown(f"### ❌ Error: Server returned {res.status_code}")
        except Exception as e:
            car_status_placeholder.markdown(f"### ❌ Exception: {e}")

        time.sleep(1)  # Small pause to reduce CPU load

# Run app
if __name__ == "__main__":
    main()
