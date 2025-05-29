import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils  # Added missing import
import dlib
import time
import streamlit as st
import requests
from threading import Thread
import playsound

# ---------------- Streamlit UI Setup ----------------
st.set_page_config(layout="wide")
st.title("🚗 Smart Driver & Vehicle Monitoring Dashboard")

col1, col2 = st.columns(2)

status_placeholder = col1.empty()
frame_placeholder = col1.empty()
car_status_placeholder = col2.empty()

# ---------------- Load Models ----------------
try:
    road_rage_model = load_model('Models/road_rage_detection_model2.h5')
    mood_model = load_model('Models/MoodDetection.keras')
    drowsiness_predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
    face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("❌ Error: Haar cascade not loaded. Check path.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ---------------- Constants ----------------
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
ALARM_PATH = "Alert.wav"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
driver_safe = True  # Global safety flag

# ---------------- Helper Functions ----------------
def sound_alarm(path):
    try:
        playsound.playsound(path)
    except Exception as e:
        print(f"Error playing sound: {e}")

def send_vehicle_status(status):
    try:
        response = requests.post("http://localhost:5000/ai_status", json={"status": status}, timeout=2)
        print(f"Sent vehicle status {status}, response: {response.status_code}")
    except Exception as e:
        print(f"Error sending vehicle status: {e}")

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

def preprocess_road_rage_frame(frame, target_size=(150, 150)):
    frame_resized = cv2.resize(frame, target_size)
    return frame_resized / 255.0

# ---------------- Main Function ----------------
def main():
    global driver_safe
    frame_buffer = []
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0

    try:
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    except Exception as e:
        st.error(f"❌ Error initializing video stream: {e}")
        st.stop()

    while True:
        try:
            frame = vs.read()
            if frame is None:
                st.warning("⚠️ No frame captured from video stream.")
                continue

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            driver_safe = True  # Reset safety flag each frame

            # ----- Drowsiness Detection -----
            rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            status_text = ""

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
                            Thread(target=sound_alarm, args=(ALARM_PATH,), daemon=True).start()
                        status_text += "### 🛑 DROWSINESS ALERT!\n"
                        driver_safe = False
                else:
                    COUNTER = 0
                    alarm_status = False

                if distance > YAWN_THRESH:
                    status_text += "### 🛑 Yawn Alert!\n"
                    if not alarm_status2 and not saying:
                        alarm_status2 = True
                        Thread(target=sound_alarm, args=(ALARM_PATH,), daemon=True).start()
                    driver_safe = False
                else:
                    alarm_status2 = False

            # ----- Mood Detection -----
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                preprocessed_face = preprocess_frame(face_roi)
                predictions = mood_model.predict(preprocessed_face, verbose=0)
                emotion = emotion_labels[np.argmax(predictions)]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                status_text += f"### 😊 Mood Detected: {emotion}\n"

                if emotion == "Angry":
                    driver_safe = False

            # ----- Road Rage Detection -----
            frame_buffer.append(preprocess_road_rage_frame(frame))
            if len(frame_buffer) == 30:
                input_data = np.array(frame_buffer).reshape((1, 30, 150, 150, 3))
                prediction = road_rage_model.predict(input_data, verbose=0)[0][0]
                label = "Violence" if prediction > 0.5 else "No Violence"
                color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
                cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                status_text += f"### 🚨 Road Rage: {label}\n"
                frame_buffer.pop(0)
            #     if label == "Violence":
            #         driver_safe = False

            # Update Streamlit UI
            status_placeholder.markdown(status_text if status_text else "### ✅ No Alerts")
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # ----- Vehicle Control -----
            send_vehicle_status(1 if driver_safe else 0)

            # ----- Fetch Vehicle and AI Status -----
            try:
                res = requests.get("http://localhost:5000/full-status", timeout=2)
                if res.ok:
                    data = res.json()
                    ai = data.get("ai_control_status")
                    vehicle = data.get("vehicle_reported_status")

                    if ai == 1 and vehicle == 1:
                        car_status_placeholder.markdown("### ✅ Vehicle Running")
                    elif ai == 1 and vehicle == 0:
                        car_status_placeholder.markdown("### 💥 Emergency Stop due to Sudden Obstacle Detection")
                    elif ai == 0 and vehicle == 1:
                        car_status_placeholder.markdown("### 🚫 Vehicle Stopped by the Driver Monitor System")
                    else:
                        car_status_placeholder.markdown("### ❌ Vehicle Stopped (Both systems issued STOP)")
                else:
                    car_status_placeholder.markdown(f"### ❌ Server Error: {res.status_code}")
            except Exception as e:
                car_status_placeholder.markdown(f"### ❌ Connection Error: {e}")

        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

        time.sleep(0.1)  # Control frame rate to avoid overwhelming Streamlit

    vs.stop()

# Run app
if __name__ == "__main__":
    main()