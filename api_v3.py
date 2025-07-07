from flask import Flask, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)

# MQTT Broker Config
MQTT_BROKER = "192.168.1.100"  # Or IP like "192.168.43.60"
MQTT_PORT = 1883
TOPIC_CONTROL = "robot/control"
TOPIC_STATUS = "robot/status"

# 🧠 Current AI control status (1 = allow, 0 = block)
ai_status = {"status": 1}

# 🚗 Latest ESP status update
vehicle_status = {"status": "CLEAR"}  # CLEAR or OBSTACLE

# 🛰️ Set up MQTT
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print("✅ Connected to MQTT Broker:", MQTT_BROKER)
    client.subscribe(TOPIC_STATUS)

def on_message(client, userdata, msg):
    global vehicle_status
    decoded = msg.payload.decode()
    print(f"📡 Received from ESP → {msg.topic}: {decoded}")
    if decoded in ["OBSTACLE", "CLEAR"]:
        vehicle_status["status"] = decoded

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# 🔁 AI sets whether vehicle can move
@app.route('/ai_status', methods=['POST'])
def update_ai_status():
    global ai_status
    data = request.get_json()
    print("📬 Received AI status:", data)

    if 'status' in data and data['status'] in [0, 1]:
        ai_status["status"] = data['status']
        command = "BLOCK" if data['status'] == 0 else "ALLOW"
        mqtt_client.publish(TOPIC_CONTROL, command)
        return jsonify({"status": "✅ Command sent to ESP", "command": command})
    else:
        return jsonify({"error": "❌ Invalid status value"}), 400

# 📊 Dashboard or monitoring tool can query both statuses
@app.route('/full-status', methods=['GET'])
def get_status():
    return jsonify({
        "ai_control_status": ai_status["status"],
        "vehicle_status": vehicle_status["status"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
