# Driver_Assistance_System-Final_Project

An intelligent driver and vehicle monitoring system built using **Deep Learning**, **Computer Vision**, and **IoT** that ensures real-time driver safety, mood-based alerts, road rage detection, and autonomous vehicle control via sensors and AI logic.

---

## 🧠 Features

### 👀 Driver Monitoring

* **Drowsiness Detection** (via Eye Aspect Ratio + Lip Distance)
* **Mood Detection** (CNN on FER2013 dataset)
* **Road Rage Detection** (3D CNN on video input)
* **Audio/Visual Alerts** on detecting danger signs

### 🐘 Wildlife Detection

* YOLOv8-based thermal object detection for **preventing roadkill**
* Alerts on animal detection in low-light/night conditions

### 🔧 Autonomous Vehicle Control

* ESP8266-based system using MQTT
* Obstacle detection with Ultrasonic + IR sensors
* Real-time decision: **Allow / Block** movement based on AI + sensors

### 🌐 Streamlit Dashboard + Flask API

* Live webcam monitoring + visual feedback
* Vehicle control API integrated with MQTT
* UI reflects real-time vehicle + AI status

---

## 🗂️ Project Structure

```bash
.
├── main_v3.py             # Streamlit UI + AI monitoring
├── api_v3.py              # Flask API + MQTT for ESP8266 control
├── Models/                # Folder for trained models (mood, drowsiness, road rage, etc.)
├── Alert.wav              # Sound file for alerts
```

---

## 🚀 How to Run

### 1. Clone this repository

```bash
git clone https://github.com/your-username/ai-car-assistant.git
cd ai-car-assistant
```

### 2. Download Pre-trained Models

🔗 [Click here to download all models (Google Drive)](https://drive.google.com/file/d/1i6CujHQCpUScewVtyDT9nJ5VuS49BwkG/view?usp=sharing)

Extract the ZIP into the `Models/` directory.

```bash
mkdir Models
# Then move the extracted files here
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Make sure you also have:

* OpenCV
* Dlib
* TensorFlow & Keras
* PyTorch
* Imutils
* Streamlit
* Paho-MQTT

💡 **Note**: You’ll need to install dlib with CMake support.

### 4. Start Flask API

```bash
python api_v3.py
```

This starts the backend API on `http://localhost:5000`.

### 5. Run Streamlit App

```bash
streamlit run main_v3.py
```

The dashboard opens in your browser.

---

## ⚙️ ESP8266 Setup

* MQTT Broker IP in `api_v3.py` must match your ESP device's network
* ESP sends `"CLEAR"` / `"OBSTACLE"` status
* Flask sends `"ALLOW"` / `"BLOCK"` commands back based on AI decisions

---

## 📷 System Overview

| Module          | Tech Used                     |
| --------------- | ----------------------------- |
| Drowsiness      | OpenCV, Dlib, EAR-based logic |
| Mood Detection  | Keras CNN (FER2013 dataset)   |
| Road Rage       | PyTorch R3D-18 model          |
| Wildlife        | YOLOv8 (custom-trained)       |
| Vehicle Control | ESP8266 + MQTT + Sensors      |
| Dashboard       | Streamlit                     |
| Backend API     | Flask + MQTT                  |

---

## 📸 Sample Screenshots

* Vehicle Stopped due to Drowsiness 😴
* Road Rage Alert 🛑
* Mood Detected: Angry 😠
* Animal Detected: Emergency Brake 🐶

---

## 🏁 Future Improvements

* Add multilingual voice command support
* Integrate with Spotify for emotion-based music therapy
* Expand wildlife detection with drone-based FLIR inputs
* Deploy to Jetson Nano or Raspberry Pi for in-car usage

---

## 📚 Credits

Built by **Divyanshu Kumar**
B.Tech in Computer Science & Engineering
Dumka Engineering College, Jharkhand
Guided by: **Dr. Amit Kumar Pramanik**


