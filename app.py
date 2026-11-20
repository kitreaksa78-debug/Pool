from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        # ១. ទទួលទិន្នន័យពី App
        data = request.get_json(force=True)
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

        h, w = img.shape[:2]

        # ២. កាត់យកតែតំបន់តុប៊ីយ៉ា (កាត់ UI ខាងលើចោល និងទុកផ្នែកខាងក្រោមឱ្យធំដើម្បីកុំឱ្យបាត់បាល់ស)
        roi_y_start = int(h * 0.10) 
        roi_y_end = int(h * 0.95)   
        roi_x_start = int(w * 0.05)
        roi_x_end = int(w * 0.95)
        
        roi_img = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # ៣. រកបាល់ស (Cue Ball) ដោយប្រើ HSV Color Mask
        # កែតម្រូវពណ៌ឱ្យធូរជាងមុន ដើម្បីកុំឱ្យបាត់បាល់ពេលមានស្រមោល
        lower_white = np.array([0, 0, 160]) 
        upper_white = np.array([180, 60, 255]) 
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cue_ball_x, cue_ball_y = None, None
        if contours:
            # តម្រៀបរង្វង់ពីធំទៅតូច រួចយករង្វង់ធំជាងគេដែលជាបាល់ស
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 80: # ទំហំអប្បបរមានៃបាល់
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cue_ball_x = int(M["m10"] / M["m00"]) + roi_x_start
                        cue_ball_y = int(M["m01"] / M["m00"]) + roi_y_start
                        break # បានបាល់សហើយ ឈប់រក

        # ៤. រកបាល់គោលដៅ (Target Ball) ដោយប្រើ HoughCircles
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.medianBlur(gray_roi, 5)
        
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=25, minRadius=8, maxRadius=20
        )

        target_x, target_y = None, None
        if circles is not None and cue_ball_x is not None:
            circles = np.round(circles[0, :]).astype("int")
            min_dist = float('inf')
            
            for (x, y, r) in circles:
                real_x = x + roi_x_start
                real_y = y + roi_y_start
                
                # គណនាចម្ងាយពីបាល់ស ទៅបាល់ដែលរកឃើញ
                dist = math.hypot(real_x - cue_ball_x, real_y - cue_ball_y)
                
                # ប្រសិនបើវាជាបាល់ផ្សេង ហើយនៅជិតជាងគេ
                if 30 < dist < min_dist: 
                    min_dist = dist
                    target_x, target_y = real_x, real_y

        # ៥. គណនាមុំ និងផ្ញើលេខត្រឡប់ទៅ App វិញ
        if cue_ball_x is not None and target_x is not None:
            # គណនាមុំពីបាល់ស ទៅបាល់គោលដៅ
            angle = math.atan2(target_y - cue_ball_y, target_x - cue_ball_x)
            
            # គូរបន្ទាត់បន្តពីបាល់គោលដៅ ទៅមុខ 400 ភីកសែល
            endX = int(target_x + 400 * math.cos(angle))
            endY = int(target_y + 400 * math.sin(angle))
            
            return jsonify({
                "status": "success",
                "startX": cue_ball_x,
                "startY": cue_ball_y,
                "endX": endX,
                "endY": endY
            })
            
        elif cue_ball_x is not None:
            # ប្រសិនបើរកឃើញតែបាល់ស គូរបន្ទាត់សាកល្បងទៅស្តាំ
            return jsonify({
                "status": "success",
                "startX": cue_ball_x,
                "startY": cue_ball_y,
                "endX": cue_ball_x + 400,
                "endY": cue_ball_y
            })
        
        # បើរកមិនឃើញបាល់សទាល់តែសោះ
        return jsonify({"status": "error", "message": "Could not find cue ball"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
