import os
import sys

# Fix for PyTorch-Streamlit compatibility issue
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Monkey patch to prevent Streamlit from inspecting torch.classes
import streamlit.watcher.local_sources_watcher
original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths

def patched_get_module_paths(module):
    if hasattr(module, '__name__') and 'torch' in str(module.__name__):
        return []
    return original_get_module_paths(module)

streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras.models import load_model
from torchvision import models
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import time
import streamlit as st
import requests
from threading import Thread
import playsound
import logging

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("🚗 Smart Driver & Vehicle Monitoring Dashboard")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
ALARM_PATH = "Alert.wav"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FRAME_BUFFER_SIZE = 16
INPUT_SHAPE = (3, 16, 112, 112)
MODEL_PATH = "Models/road_rage_r3d18_model.pth"

# Load Models
@st.cache_resource
def load_pytorch_model():
    try:
        model = models.video.r3d_18(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
        return model
    except Exception as e:
        logging.error(f"Failed to load PyTorch model: {e}")
        st.error(f"PyTorch model loading failed: {e}")
        return None

try:
    mood_model = load_model('Models/MoodDetection.keras')
    drowsiness_predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
    face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    pt_model = load_pytorch_model()
    if face_cascade.empty():
        st.error("❌ Haar cascade not loaded.")
        st.stop()
    if pt_model is None:
        st.error("❌ PyTorch road rage model failed to load.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Helper Functions
def sound_alarm(path):
    try:
        playsound.playsound(path)
    except Exception as e:
        logging.error(f"Sound error: {e}")

def send_vehicle_status(status):
    try:
        requests.post("http://localhost:5000/ai_status", json={"status": status}, timeout=2)
    except Exception as e:
        logging.error(f"Vehicle status send error: {e}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    try:
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
        rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
        return (leftEAR + rightEAR) / 2.0
    except Exception as e:
        logging.error(f"EAR calculation error: {e}")
        return 0.0

def lip_distance(shape):
    try:
        top_lip = np.concatenate((shape[50:53], shape[61:64]))
        low_lip = np.concatenate((shape[56:59], shape[65:68]))
        return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])
    except Exception as e:
        logging.error(f"Lip distance error: {e}")
        return 0.0

def preprocess_frame(frame, size=(48, 48)):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        norm = resized / 255.0
        return np.expand_dims(np.expand_dims(norm, -1), 0)
    except Exception as e:
        logging.error(f"Frame preprocessing error: {e}")
        return None

def preprocess_frames(frame_buffer):
    try:
        if len(frame_buffer) != FRAME_BUFFER_SIZE:
            return None
        frames = np.stack(frame_buffer).astype(np.float32) / 255.0
        frames = frames.transpose(3, 0, 1, 2)
        return torch.tensor(frames).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Frame buffer preprocessing error: {e}")
        return None

def predict(model, frames):
    try:
        if frames is None:
            return None, None
        with torch.no_grad():
            prob = torch.sigmoid(model(frames)).item()
            if prob >= 0.7:
                return "Road Rage", prob
            elif 0.55 < prob < 0.7:
                return "Potential Road Rage", prob
            return "No Road Rage", prob
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None

# Streamlit UI Placeholders
st.write(f"Using device: `{device}`")
col1, col2 = st.columns(2)
status_placeholder = col1.empty()
frame_placeholder = col1.empty()
car_status_placeholder = col2.empty()

# Main Loop
def main():
    frame_buffer = []
    COUNTER = 0
    alarm_status = alarm_status2 = False

    try:
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    except Exception as e:
        st.error(f"❌ Video stream error: {e}")
        st.stop()

    run = st.checkbox("Start Webcam")
    if not run:
        vs.stop()
        return

    while run:
        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        status_text = ""
        driver_safe = True

        rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = face_utils.shape_to_np(drowsiness_predictor(gray, rect))
            ear = final_ear(shape)
            distance = lip_distance(shape)

            # Draw eye and lip contours
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=sound_alarm, args=(ALARM_PATH,), daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status_text += "### 🛑 DROWSINESS ALERT!\n"
                    driver_safe = False
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                status_text += "### 🛑 Yawn Alert!\n"
                if not alarm_status2:
                    alarm_status2 = True
                    Thread(target=sound_alarm, args=(ALARM_PATH,), daemon=True).start()
                    driver_safe = False
            else:
                alarm_status2 = False

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            processed = preprocess_frame(roi)
            if processed is not None:
                emotion = emotion_labels[np.argmax(mood_model.predict(processed, verbose=0))]
                # Draw rectangle around face and emotion label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                status_text += f"### 😊 Mood Detected: {emotion}\n"
                if emotion == "Angry":
                    driver_safe = False

        frame_buffer.append(cv2.resize(frame_rgb, (112, 112)))
        if len(frame_buffer) == FRAME_BUFFER_SIZE:
            tensor = preprocess_frames(frame_buffer)
            pred, prob = predict(pt_model, tensor)
            if pred:
                # Display road rage detection on video
                color = (0, 0, 255) if pred == "Road Rage" else (255, 165, 0) if pred == "Potential Road Rage" else (0, 255, 0)
                cv2.putText(frame, f"{pred} ({prob:.2f})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                status_text += f"### 🚨 Road Rage: {pred} (conf: {prob:.2f})\n"
                if pred != "No Road Rage":
                    driver_safe = False
            frame_buffer.pop(0)

        status_placeholder.markdown(status_text or "### ✅ No Alerts")
        frame_placeholder.image(frame_rgb, channels="RGB")
        send_vehicle_status(1 if driver_safe else 0)

        try:
            res = requests.get("http://localhost:5000/full-status", timeout=2)
            if res.ok:
                data = res.json()
                ai = int(data.get("ai_control_status", 0))
                vehicle = 1 if data.get("vehicle_status") == "CLEAR" else 0
                if ai and vehicle:
                    car_status_placeholder.markdown("### ✅ Vehicle Running")
                elif ai:
                    car_status_placeholder.markdown("### 💥 Emergency Stop: Obstacle")
                elif vehicle:
                    car_status_placeholder.markdown("### 🚫 Stop by Driver Monitor")
                else:
                    car_status_placeholder.markdown("### ❌ Vehicle Stopped")
        except:
            car_status_placeholder.markdown("### ⚠️ Status Fetch Error")

main()