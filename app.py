from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from datetime import datetime

app = Flask(__name__)

# Mediapipe and utility setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables
count = 0
target_count = 0
position = None
exercise_started = False
feedback_message = "Begin PostureTraining!"
start_time = None
last_rep_time = None
current_user = None
exercise = None

# Database file
DATABASE_FILE = 'users.json'

# Load users from JSON file
def load_users():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save users to JSON file
def save_users(users):
    with open(DATABASE_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

# Generate video frames
def generate_frames():
    global count, position, exercise_started, feedback_message, start_time, last_rep_time, exercise
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and exercise_started and exercise:
            landmarks = results.pose_landmarks.landmark

            joints = exercise["joints"]
            coords = [
                [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                 landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                for joint in joints
            ]

            angle = calculate_angle(*coords)

            # PostureTraining counting and feedback logic
            if angle > exercise["target_angle"] + exercise["threshold"]:
                position = "up"
            if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                position = "down"
                count += 1

                # Calculate time for the repetition
                current_time = time.time()
                if last_rep_time:
                    rep_time = current_time - last_rep_time
                    if exercise["optimal_speed_range"][0] <= rep_time <= exercise["optimal_speed_range"][1]:
                        feedback_message = "Good speed! Keep going."
                    elif rep_time < exercise["optimal_speed_range"][0]:
                        feedback_message = "Too fast! Slow down."
                    else:
                        feedback_message = "Too slow! Speed up."
                last_rep_time = current_time

                # Start timer for the first rep
                if count == 1:
                    start_time = current_time

            # Provide feedback based on angle
            if angle < exercise["target_angle"] - exercise["threshold"]:
                feedback_message = "Raise your knee higher!"
            elif angle > exercise["target_angle"] + exercise["threshold"]:
                feedback_message = "Lower your knee slightly."

            # Draw feedback on the frame
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {count}/{target_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, feedback_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Stop exercise if target count is reached
            if count >= target_count:
                exercise_started = False
                total_time = current_time - start_time
                feedback_message = f"PostureTraining Complete! Total time: {total_time:.2f}s"
                cv2.putText(image, feedback_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save session data
                if current_user:
                    users = load_users()
                    if current_user in users:
                        session_data = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "count": count,
                            "total_time": total_time,
                            "average_speed": total_time / count if count > 0 else 0
                        }
                        users[current_user]["sessions"].append(session_data)
                        save_users(users)

        # Encode the frame and send to the frontend
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        global current_user
        username = request.form['username']
        users = load_users()
        if username in users:
            current_user = username
            return redirect(url_for('select_exercise'))
        else:
            return render_template('login.html', error="User not found. Please register.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global current_user
        username = request.form['username']
        age = request.form['age']
        users = load_users()
        if username in users:
            return render_template('register.html', error="Username already exists.")
        users[username] = {"age": age, "sessions": []}
        save_users(users)
        current_user = username
        return redirect(url_for('select_exercise'))
    return render_template('register.html')

@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    global exercise
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        if exercise_name == "knee_raises":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 60,  # Ideal angle for knee raised to hip level
                "threshold": 15,
                "optimal_speed_range": (1.0, 2.5)  # Optimal time in seconds for one rep
            }
        elif exercise_name == "squats":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 90,  # Ideal angle for squat
                "threshold": 15,
                "optimal_speed_range": (2.0, 4.0)  # Optimal time in seconds for one rep
            }
        return redirect(url_for('training'))
    return render_template('select_exercise.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    global count, target_count, feedback_message
    return jsonify({'count': count, 'target': target_count, 'feedback': feedback_message})

@app.route('/set_target', methods=['POST'])
def set_target():
    global target_count, count, exercise_started, feedback_message, start_time, last_rep_time
    data = request.json
    target_count = int(data.get('target', 0))
    count = 0
    exercise_started = True
    feedback_message = "Begin PostureTraining!"
    start_time = None
    last_rep_time = None
    return jsonify({'success': True, 'target': target_count})

if __name__ == "__main__":
    app.run(debug=True)