from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "API is running!"

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    # នេះជាកន្លែងសម្រាប់ដាក់កូដ OpenCV របស់អ្នក
    return jsonify({
        "status": "success", 
        "message": "Received image successfully",
        "startX": 150, 
        "startY": 250, 
        "endX": 600, 
        "endY": 400
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
