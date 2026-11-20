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
        data = request.get_json(force=True)
        image_data = data.get('image')

        if not image_data:
            return jsonify({"status": "error", "message": "No image"}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        h, w = img.shape[:2]

        # កំណត់ទីតាំងរន្ធទាំង 6 (សម្រាប់អេក្រង់ 1600x720)
        pockets = [
            (110, 110), (800, 80), (1490, 110),    # រន្ធខាងលើ
            (110, 610), (800, 640), (1490, 610)    # រន្ធខាងក្រោម
        ]

        # ១. កាត់យកតំបន់តុប៊ីយ៉ា
        roi_y_start = int(h * 0.12) 
        roi_y_end = int(h * 0.90)   
        roi_x_start = int(w * 0.08)
        roi_x_end = int(w * 0.92)
        
        roi_img = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # ២. រកបាល់ស (Cue Ball) - បន្ថែមការពិនិត្យរាងមូល ដើម្បីកុំឱ្យចាប់ដំបង
        lower_white = np.array([0, 0, 180]) 
        upper_white = np.array([180, 45, 255]) 
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cue_ball_x, cue_ball_y = None, None
        if contours:
            # តម្រៀបរង្វង់ពីធំទៅតូច
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 100: # បាល់សត្រូវមានទំហំល្មម
                    # ពិនិត្យមើលរាង (Aspect Ratio) ថាតើវាមូលដូចបាល់ឬវែងដូចដំបង
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    aspect_ratio = float(w_box) / h_box
                    if 0.8 <= aspect_ratio <= 1.2: # បើរាងមូលស្មើៗគ្នា
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cue_ball_x = int(M["m10"] / M["m00"]) + roi_x_start
                            cue_ball_y = int(M["m01"] / M["m00"]) + roi_y_start
                            break

        # ៣. រកបាល់គោលដៅទាំងអស់
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.medianBlur(gray_roi, 5)
        
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
            param1=50, param2=25, minRadius=8, maxRadius=20
        )

        target_x, target_y = None, None
        best_endX, best_endY = None, None

        if circles is not None and cue_ball_x is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # រកបាល់ដែលនៅជិតរន្ធជាងគេ ដើម្បីវាយឱ្យចូល
            shortest_dist_to_pocket = float('inf')
            
            for (x, y, r) in circles:
                real_x = x + roi_x_start
                real_y = y + roi_y_start
                
                # កុំឱ្យវាចាប់បាល់សខ្លួនឯង
                if math.hypot(real_x - cue_ball_x, real_y - cue_ball_y) < 30:
                    continue

                # រកមើលថាតើបាល់នេះនៅជិតរន្ធណាបំផុត
                for (px, py) in pockets:
                    dist_to_pocket = math.hypot(real_x - px, real_y - py)
                    if dist_to_pocket < shortest_dist_to_pocket:
                        shortest_dist_to_pocket = dist_to_pocket
                        target_x, target_y = real_x, real_y
                        best_endX, best_endY = px, py # គូរបន្ទាត់ទៅរន្ធនោះតែម្តង

        # ៤. គណនាមុំ និងផ្ញើលទ្ធផល
        if cue_ball_x is not None and target_x is not None and best_endX is not None:
            # គណនាមុំពីបាល់គោលដៅ ទៅរន្ធ
            angle_to_pocket = math.atan2(best_endY - target_y, best_endX - target_x)
            
            # គណនាទីតាំង "Ghost Ball" (ចំណុចដែលបាល់សត្រូវប៉ះ)
            ghost_x = int(target_x - 30 * math.cos(angle_to_pocket))
            ghost_y = int(target_y - 30 * math.sin(angle_to_pocket))
            
            return jsonify({
                "status": "success",
                "startX": cue_ball_x,
                "startY": cue_ball_y,
                "endX": ghost_x,
                "endY": ghost_y,
                "targetX": best_endX,  # បន្ថែមទីតាំងរន្ធដើម្បីឱ្យ App គូរបន្ទាត់ទី២
                "targetY": best_endY
            })
            
        return jsonify({"status": "error", "message": "Could not find a valid shot"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
