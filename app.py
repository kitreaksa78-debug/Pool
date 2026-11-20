from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)

# --- កំណត់ប៉ារ៉ាម៉ែត្រសម្រាប់ការគណនា (អាចត្រូវកែតម្រូវតាមទំហំអេក្រង់) ---
BALL_RADIUS = 12       # កាំនៃបាល់ (គិតជាភីកសែល)
HOLE_RADIUS = 25       # កាំនៃរន្ធ
# ទីតាំងរន្ធទាំង 6 (ឧបមាថាតុមានទំហំ 800x400 ភីកសែល សម្រាប់គំរូ)
# អ្នកត្រូវកែតម្រូវលេខទាំងនេះឱ្យត្រូវនឹងរូបភាពពិតរបស់អ្នក
POCKETS = [
    (50, 50), (400, 50), (750, 50),       # រន្ធខាងលើ (កំណត់ពីឆ្វេងទៅស្តាំ)
    (50, 350), (400, 350), (750, 350)      # រន្ធខាងក្រោម
]

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

        # ១. បំប្លែង Base64 ទៅជារូបភាព OpenCV
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_draw = img.copy()

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        # ២. រកបាល់ស (Cue Ball) ដោយប្រើ HSV Color Masking
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # ជួរពណ៌សសម្រាប់បាល់ស
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # សម្អាតសំរាមជុំវិញ
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

        contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cue_ball_pos = None
        if contours_white:
            # យករង្វង់ធំបំផុតដែលជាបាល់ស
            largest_white = max(contours_white, key=cv2.contourArea)
            M = cv2.moments(largest_white)
            if M["m00"] != 0:
                cue_ball_pos = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # ៣. រកបាល់គោលដៅ (Target Ball) ដោយប្រើ HoughCircles
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # កាត់តែតំបន់តុប៊ីយ៉ា (ដើម្បីកាត់បន្ថយការភ័ណ្ឌចាប់ UI)
        h, w = gray.shape
        # កែតម្រូវតំបន់កាត់ (crop) នេះតាមទំហំអេក្រង់ហ្គេមរបស់អ្នក
        crop_gray = gray[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
        
        circles = cv2.HoughCircles(
            crop_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=BALL_RADIUS-5, maxRadius=BALL_RADIUS+5
        )

        target_ball_pos = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            min_dist = float('inf')
            
            for (x, y, r) in circles:
                # បំប្លែងកូអរដោនេក្រោយកាត់រូប មកទីតាំងដើមវិញ
                real_x = x + int(w*0.1)
                real_y = y + int(h*0.15)
                
                if cue_ball_pos is not None:
                    dist = math.hypot(real_x - cue_ball_pos[0], real_y - cue_ball_pos[1])
                    # ប្រសិនបើវាជាបាល់ផ្សេង ហើយនៅជិតជាងគេ
                    if 30 < dist < min_dist: 
                        min_dist = dist
                        target_ball_pos = (real_x, real_y)

        # ៤. គណនាគន្លង និងគូរបន្ទាត់
        if cue_ball_pos is not None and target_ball_pos is not None:
            # គូរបន្ទាត់ពីបាល់ស ទៅបាល់គោលដៅ (គន្លងវាយ)
            cv2.line(img_draw, cue_ball_pos, target_ball_pos, (0, 0, 255), 3)  # បន្ទាត់ក្រហម
            
            # គណនាទិសដៅបន្តពីបាល់គោលដៅ (ដើម្បីបង្ហាញថាវានឹងទៅណា)
            angle = math.atan2(target_ball_pos[1] - cue_ball_pos[1], target_ball_pos[0] - cue_ball_pos[0])
            # គូរបន្ទាត់បន្តប្រវែង 300 ភីកសែល
            end_x = int(target_ball_pos[0] + 300 * math.cos(angle))
            end_y = int(target_ball_pos[1] + 300 * math.sin(angle))
            cv2.line(img_draw, target_ball_pos, (end_x, end_y), (255, 0, 0), 2)  # បន្ទាត់ពណ៌ខៀវ
            
            # គូររង្វង់តូចៗសម្គាល់ទីតាំងបាល់
            cv2.circle(img_draw, cue_ball_pos, BALL_RADIUS, (0, 255, 0), 2)
            cv2.circle(img_draw, target_ball_pos, BALL_RADIUS, (0, 255, 255), 2)

        elif cue_ball_pos is not None:
            # ប្រសិនបើរកឃើញតែបាល់ស គូរបន្ទាត់សាកល្បងទៅមុខ
            cv2.line(img_draw, cue_ball_pos, (cue_ball_pos[0] + 400, cue_ball_pos[1]), (0, 0, 255), 3)

        # ៥. បំប្លែងរូបភាពត្រឡប់ទៅវិញជា Base64
        _, buffer = cv2.imencode('.jpg', img_draw)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "image": "data:image/jpeg;base64," + processed_image_base64
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
