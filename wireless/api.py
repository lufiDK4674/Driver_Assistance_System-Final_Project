from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🧠 AI-controlled vehicle status
vehicle_status = {"status": 1}  # 1 = running, 0 = stop

# ✅ AI updates vehicle status
@app.route('/data', methods=['POST'])
def ai_update_status():
    global vehicle_status
    data = request.get_json()
    print("📬 AI Update Received:", data)
    if 'status' in data and data['status'] in [0, 1]:
        vehicle_status["status"] = data['status']
        return jsonify({"status": "AI updated vehicle status successfully"})
    else:
        return jsonify({"error": "Invalid status from AI"}), 400

# 🚗 Vehicle polls status
@app.route('/get', methods=['GET'])
def vehicle_get_status():
    print("📡 Vehicle requested current status")
    print("📤 Sending to vehicle:", vehicle_status)
    return jsonify(vehicle_status)  # Always send status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
