from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64
import math  # បន្ថែមមកដើម្បីគណនាមុំ

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

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        # ១. រកបាល់ស
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180]) 
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cue_ball_x = int(M["m10"] / M["m00"])
                cue_ball_y = int(M["m01"] / M["m00"])
                
                # ២. គណនាមុំ (ឧបមាថាយើងចង់វាយទៅស្តាំដៃបន្តិច)
                # ក្នុងកម្មវិធីពិត អ្នកត្រូវរកទីតាំងបាល់គោលដៅ (Target) ជាមុនសិន
                target_x = cue_ball_x + 400 
                target_y = cue_ball_y + 100 
                
                # គណនាមុំដោយប្រើ math.atan2
                angle = math.atan2(target_y - cue_ball_y, target_x - cue_ball_x)
                
                # គណនាប្រវែងបន្ទាត់ (ឧទាហរណ៍៖ ទៅមុខ ៦០០ ភីកសែល)
                line_length = 600
                endX = int(cue_ball_x + line_length * math.cos(angle))
                endY = int(cue_ball_y + line_length * math.sin(angle))
                
                # ៣. គូរបន្ទាត់លើរូបភាពផ្ទាល់ (ដើម្បីឱ្យ Website បង្ហាញលទ្ធផល)
                cv2.line(img, (cue_ball_x, cue_ball_y), (endX, endY), (0, 0, 255), 4) # បន្ទាត់ក្រហម
                
                # បំប្លែងរូបភាពដែលគូររួច ទៅជា Base64 ដើម្បីផ្ញើទៅ Website
                _, buffer = cv2.imencode('.jpg', img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    "status": "success",
                    "image": "data:image/jpeg;base64," + processed_image_base64
                })
        
        return jsonify({"status": "error", "message": "Could not find the cue ball"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
