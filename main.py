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

# Load models
print("-> Loading models...")
road_rage_model = load_model('Models/road_rage_detection_model2.h5')
mood_model = load_model('Models/MoodDetection.keras')
drowsiness_predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Haar cascade file not loaded. Check the file path.")
    exit()
# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Alarm function
def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        playsound.playsound(path)
    if alarm_status2:
        saying = True
        playsound.playsound(path)
        saying = False

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Final EAR calculation
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

# Lip distance calculation
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# Preprocess frame for mood detection
def preprocess_frame(frame, target_size=(48, 48)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, target_size)
    normalized_frame = resized_frame / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    return np.expand_dims(expanded_frame, axis=0)

# Main function
def main():
    frame_buffer = []
    global alarm_status, alarm_status2, COUNTER

    print("-> Starting Video Stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drowsiness detection
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = drowsiness_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        t = Thread(target=sound_alarm, args=("Alert.wav",))
                        t.daemon = True
                        t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    t = Thread(target=sound_alarm, args=("Alert.wav",))
                    t.daemon = True
                    t.start()
            else:
                alarm_status2 = False

        # Mood detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_frame(face_roi)
            predictions = mood_model.predict(preprocessed_face)
            emotion = emotion_labels[np.argmax(predictions)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Road rage detection

        frame_resized = cv2.resize(frame, (150, 150))
        frame_buffer.append(frame_resized)

        # Ensure the buffer contains exactly 30 frames
        if len(frame_buffer) == 30:
            input_data = np.array(frame_buffer).reshape((1, 30, 150, 150, 3)) / 255.0
            prediction = road_rage_model.predict(input_data)[0][0]
            label = "Violence" if prediction > 0.6 else "No Violence"
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Remove the oldest frame to maintain the buffer size
            frame_buffer.pop(0)

        # Display the frame
        cv2.imshow("Driver Assistance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()