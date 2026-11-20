from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        # បំប្លែង Base64 ទៅជារូបភាព
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        # រកកូនប៊ីយ៉ាស (បាល់ពណ៌ស) ដោយប្រើ HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180]) 
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # សម្អាតសំរាមជុំវិញ
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # រកទីតាំងកូនប៊ីយ៉ាស
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cue_ball_x = int(M["m10"] / M["m00"])
                cue_ball_y = int(M["m01"] / M["m00"])
                
                # គូរបន្ទាត់ពីបាល់ស ទៅមុខ ៥០០ ភីកសែល (ជាការសាកល្បង)
                target_x = cue_ball_x + 500
                target_y = cue_ball_y
                
                return jsonify({
                    "status": "success",
                    "startX": cue_ball_x,
                    "startY": cue_ball_y,
                    "endX": target_x,
                    "endY": target_y
                })
        
        return jsonify({"status": "error", "message": "Could not find the cue ball"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
