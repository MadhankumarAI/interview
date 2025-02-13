# import threading
# import time
# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import dlib
# from deepface import DeepFace

# app = Flask(__name__)

# cap = None
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is present

# confidence_scores = []  # Store confidence levels
# emotion_scores = {  # Store count of detected emotions
#     "happy": 0, "neutral": 0, "surprise": 0, 
#     "sad": 0, "fear": 0, "angry": 0, "disgust": 0
# }
# lock = threading.Lock()  # Thread safety for shared resources

# # Camera properties
# frame_width, frame_height = 640, 480
# focal_length, center = frame_width, (frame_width / 2, frame_height / 2)

# # Camera matrix (no distortion assumed)
# camera_matrix = np.array([
#     [focal_length, 0, center[0]],
#     [0, focal_length, center[1]],
#     [0, 0, 1]
# ], dtype="double")

# # Face orientation detection
# def detect_face_not_looking(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)
#         image_points = np.array([
#             (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
#             (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
#             (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
#             (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
#             (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
#         ], dtype="double")

#         centroid_x, centroid_y = np.mean(image_points[:, 0]), np.mean(image_points[:, 1])
#         frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
#         distance_x, distance_y = abs(centroid_x - frame_center_x), abs(centroid_y - frame_center_y)

#         threshold_x, threshold_y = 40, 40  # Lower tolerance for better detection
#         return distance_x > threshold_x or distance_y > threshold_y
#     return False

# # Emotion detection in a background thread
# def analyze_emotions():
#     global cap
#     confidence_map = {
#         "happy": 0.9, "neutral": 0.8, "surprise": 0.7,
#         "sad": 0.5, "fear": 0.4, "angry": 0.3, "disgust": 0.2
#     }
#     while True:
#         if cap and cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 try:
#                     result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
#                     if result:
#                         dominant_emotion = result[0]['dominant_emotion']
#                         confidence_score = confidence_map.get(dominant_emotion, 0.5)

#                         with lock:
#                             confidence_scores.append(confidence_score)
#                             if dominant_emotion in emotion_scores:
#                                 emotion_scores[dominant_emotion] += 1
#                 except Exception as e:
#                     print("Emotion detection error:", e)
#         time.sleep(2)  # Run every 2 seconds

# # Start background emotion detection thread
# emotion_thread = threading.Thread(target=analyze_emotions, daemon=True)
# emotion_thread.start()

# # Video streaming function
# def gen():
#     global cap
#     while cap and cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if detect_face_not_looking(frame):
#             cv2.putText(frame, "Warning: Not looking at the camera!", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         _, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('interview.html') # satwik this is the route

# @app.route('/video_feed')
# def video_feed():
#     global cap
#     if cap and cap.isOpened():
#         return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return "Camera is off. Click 'Start Camera' to begin."

# @app.route('/toggle_camera')
# def toggle_camera():
#     global cap
#     if cap is None or not cap.isOpened():
#         cap = cv2.VideoCapture(0)
#         return "Camera started" if cap.isOpened() else "Failed to start the camera."
#     else:
#         cap.release()
#         return "Camera stopped"

# @app.route('/end_meeting')
# def end_meeting():
#     global cap
#     if cap and cap.isOpened():
#         cap.release()  # Stop video streaming
    
#     total_score = sum(confidence_scores)  # Calculate total confidence score
    
#     emotion_icons = {
#         "happy": "😊", "neutral": "😐", "surprise": "😲",
#         "sad": "😢", "fear": "😨", "angry": "😠", "disgust": "🤢"
#     }

#     return render_template('meeting_ended.html', 
#                            confidence_scores=confidence_scores,
#                            emotion_scores=emotion_scores, 
#                            total_score=total_score, 
#                            emotion_icons=emotion_icons)

# @app.route('/confidence_scores')
# def get_confidence_scores():
#     with lock:
#         return {"confidence_scores": confidence_scores}

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True)
import threading
import time
from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from deepface import DeepFace

app = Flask(__name__)

cap = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is present

confidence_scores = []  # Store confidence levels
emotion_scores = {  # Store count of detected emotions
    "happy": 0, "neutral": 0, "surprise": 0, 
    "sad": 0, "fear": 0, "angry": 0, "disgust": 0
}
lock = threading.Lock()  # Thread safety for shared resources

# Face orientation detection
def detect_face_not_looking(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 1:
        return "multiple_faces"
    
    for face in faces:
        landmarks = predictor(gray, face)
        image_points = np.array([
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(30).x, landmarks.part(30).y),
            (landmarks.part(48).x, landmarks.part(48).y),
            (landmarks.part(54).x, landmarks.part(54).y)
        ], dtype="double")

        centroid_x, centroid_y = np.mean(image_points[:, 0]), np.mean(image_points[:, 1])
        frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
        distance_x, distance_y = abs(centroid_x - frame_center_x), abs(centroid_y - frame_center_y)

        threshold_x, threshold_y = 30, 50
        return distance_x > threshold_x or distance_y > threshold_y
    return False

# Emotion detection thread
def analyze_emotions():
    global cap
    confidence_map = {"happy": 0.9, "neutral": 0.8, "surprise": 0.7, "sad": 0.5, "fear": 0.4, "angry": 0.3, "disgust": 0.2}
    while True:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try:
                    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    if result:
                        dominant_emotion = result[0]['dominant_emotion']
                        confidence_score = confidence_map.get(dominant_emotion, 0.5)
                        with lock:
                            confidence_scores.append(confidence_score)
                            if dominant_emotion in emotion_scores:
                                emotion_scores[dominant_emotion] += 1
                except Exception as e:
                    print("Emotion detection error:", e)
        time.sleep(2)

emotion_thread = threading.Thread(target=analyze_emotions, daemon=True)
emotion_thread.start()

# Video streaming function
def gen():
    global cap
    while cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        warning_message = None
        
        if len(faces) > 1:
            warning_message = "Warning: Multiple faces detected!"
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        elif detect_face_not_looking(frame):
            warning_message = "Warning: Not looking at the camera!"
        
        if warning_message:
            cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('interview.html')

@app.route('/video_feed')
def video_feed():
    global cap
    if cap and cap.isOpened():
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera is off. Click 'Start Camera' to begin."

@app.route('/toggle_camera')
def toggle_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        return "Camera started" if cap.isOpened() else "Failed to start the camera."
    else:
        cap.release()
        return "Camera stopped"

@app.route('/end_meeting')
def end_meeting():
    global cap
    if cap and cap.isOpened():
        cap.release()
    
    total_score = sum(confidence_scores)
    emotion_icons = {"happy": "😊", "neutral": "😐", "surprise": "😲", "sad": "😢", "fear": "😨", "angry": "😠", "disgust": "🤢"}

    return render_template('meeting_ended.html', 
                           confidence_scores=confidence_scores,
                           emotion_scores=emotion_scores, 
                           total_score=total_score, 
                           emotion_icons=emotion_icons)

@app.route('/confidence_scores')
def get_confidence_scores():
    with lock:
        return {"confidence_scores": confidence_scores}

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
