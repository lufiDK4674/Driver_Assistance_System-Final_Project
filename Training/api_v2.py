from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🧠 AI-sent status
ai_status = {"status": 1}  # 1 = allowed to run, 0 = stop

# 🚗 Vehicle-reported status
vehicle_status = {"status": 1}  # 1 = running, 0 = stopped by IR or obstacle

# ✅ AI updates vehicle control status
@app.route('/ai_status', methods=['POST'])
def ai_update_status():
    global ai_status
    data = request.get_json()
    print("📬 AI Update Received:", data)
    if 'status' in data and data['status'] in [0, 1]:
        ai_status["status"] = data['status']
        return jsonify({"status": "✅ AI status updated successfully"})
    else:
        return jsonify({"error": "❌ Invalid status from AI"}), 400

# 🚗 Vehicle reports its own status
@app.route('/data', methods=['POST'])
def vehicle_update_status():
    global vehicle_status
    data = request.get_json()
    print("📦 Vehicle Status Update:", data)
    if 'status' in data and data['status'] in [0, 1]:
        vehicle_status["status"] = data['status']
        return jsonify({"status": "✅ Vehicle status updated successfully"})
    else:
        return jsonify({"error": "❌ Invalid status from vehicle"}), 400

# 📡 Vehicle asks if it can move (AI's control status)
@app.route('/get', methods=['GET'])
def vehicle_get_ai_status():
    print("📡 Vehicle requested AI control status")
    print("📤 Sending AI status:", ai_status)
    return jsonify(ai_status)

# 📊 Dashboard or monitoring tool can query both statuses
@app.route('/full-status', methods=['GET'])
def get_full_status():
    return jsonify({
        "vehicle_reported_status": vehicle_status["status"],
        "ai_control_status": ai_status["status"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
