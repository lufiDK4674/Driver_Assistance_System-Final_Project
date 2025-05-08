from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 👈 This enables CORS

status_message = {"message": "Waiting for signal..."}

@app.route('/get', methods=['GET'])
def get_status():
    return jsonify(status_message)

@app.route('/data', methods=['POST'])
def receive_data():
    global status_message
    data = request.get_json()
    print("📬 Message received:", data['message'])
    status_message["message"] = data['message']
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # 👈 Debug set to False
