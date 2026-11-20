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
        # ១. ទទួលរូបភាពពី App
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

        # ទំហំអេក្រង់របស់អ្នកគឺ 1600x720 (Landscape)
        h, w = img.shape[:2]

        # ---------------------------------------------------------
        # ២. កំណត់តំបន់តុប៊ីយ៉ា (Region of Interest - ROI)
        # ដើម្បីកុំឱ្យវាចាប់យក UI ខាងលើ និងខាងក្រោមធ្វើជាបាល់
        # ---------------------------------------------------------
        roi_y_start = int(h * 0.15) # កាត់ពីលើចុះមក 15%
        roi_y_end = int(h * 0.85)   # កាត់ពីក្រោមឡើង 15%
        roi_x_start = int(w * 0.10) # កាត់ពីឆ្វេងចូល 10%
        roi_x_end = int(w * 0.90)   # កាត់ពីស្តាំចូល 10%
        
        roi_img = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------
        # ៣. រកបាល់ស (Cue Ball)
        # ---------------------------------------------------------
        # ជួរពណ៌ស
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        # សម្អាតសំរាម
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cue_ball_x, cue_ball_y = None, None
        if contours:
            largest_white = max(contours, key=cv2.contourArea)
            # ប្រៀបធៀបទំហំ ដើម្បីកុំឱ្យចាប់យកអក្សរពណ៌សតូចៗ
            if cv2.contourArea(largest_white) > 200: 
                M = cv2.moments(largest_white)
                if M["m00"] != 0:
                    # បំប្លែងកូអរដោនេ ROI មកទីតាំងដើមវិញ
                    cue_ball_x = int(M["m10"] / M["m00"]) + roi_x_start
                    cue_ball_y = int(M["m01"] / M["m00"]) + roi_y_start

        # ---------------------------------------------------------
        # ៤. រកបាល់គោលដៅ (Target Ball) ដោយប្រើ HoughCircles
        # ---------------------------------------------------------
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.medianBlur(gray_roi, 5)
        
        # រករង្វង់ (កាំបាល់នៅលើអេក្រង់ 1600x720 មានទំហំប្រហែល 10-18 px)
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
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
                if 25 < dist < min_dist: 
                    min_dist = dist
                    target_x, target_y = real_x, real_y

        # ---------------------------------------------------------
        # ៥. គណនាមុំ និងផ្ញើលេខត្រឡប់ទៅវិញ
        # ---------------------------------------------------------
        if cue_ball_x is not None and target_x is not None:
            # គណនាមុំពីបាល់ស ទៅបាល់គោលដៅ
            angle = math.atan2(target_y - cue_ball_y, target_x - cue_ball_x)
            
            # គូរបន្ទាត់បន្តពីបាល់គោលដៅ ទៅមុខ 300 ភីកសែល
            endX = int(target_x + 300 * math.cos(angle))
            endY = int(target_y + 300 * math.sin(angle))
            
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
        
        # បើរកមិនឃើញអ្វីសោះ
        return jsonify({"status": "error", "message": "Could not find cue ball"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
