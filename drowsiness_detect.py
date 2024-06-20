from threading import Thread

import cv2
import dlib
import numpy as np
import pygame
from flask import Flask, Response, jsonify, render_template, request
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance

app = Flask(__name__)

# Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')
current_status_message = "Normal"

# Default thresholds and frame counts
EYE_ASPECT_RATIO_THRESHOLD = 0.20
EYE_ASPECT_RATIO_CONSEC_FRAMES = 80
MOUTH_ASPECT_RATIO_THRESHOLD = 0.55
MOUTH_ASPECT_RATIO_CONSEC_FRAMES = 30
NO_FACE_CONSEC_FRAMES = 30  # Define this variable to avoid the NameError

# Audio setting
AUDIO_ENABLED = True

# Counts no. of consecutive frames below threshold value for eye drowsiness detection
EYE_COUNTER = 0
# Counts no. of consecutive frames above threshold value for mouth open detection
MOUTH_COUNTER = 0
# Counts no. of consecutive frames with no face detection
NO_FACE_COUNTER = 0

# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Define the indexes for the outer mouth points
outer_mouth_start, outer_mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video stream initialization
vs = None

def start_camera():
    global vs
    vs = VideoStream(src=0).start()

def stop_camera():
    global vs
    if vs:
        vs.stop()

@app.route('/')
def index():
    return render_template('index.html')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])

    mar = (A + B + C) / (3 * D)
    return mar

def detect_drowsiness():
    global EYE_COUNTER, MOUTH_COUNTER, NO_FACE_COUNTER, current_status_message

    drowsy_detected = False

    while True:
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            NO_FACE_COUNTER += 1
            if NO_FACE_COUNTER >= NO_FACE_CONSEC_FRAMES:
                current_status_message = "No face detected"
                drowsy_detected = True
        else:
            NO_FACE_COUNTER = 0
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[outer_mouth_start:outer_mouth_end]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                mar = mouth_aspect_ratio(mouth)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    EYE_COUNTER += 1
                    if EYE_COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                        if AUDIO_ENABLED:
                            pygame.mixer.music.play(-1)
                        current_status_message = "You are Drowsy"
                        drowsy_detected = True
                        break  # Exit for loop once drowsiness detected
                else:
                    EYE_COUNTER = 0

                if mar > MOUTH_ASPECT_RATIO_THRESHOLD:
                    MOUTH_COUNTER += 1
                    if MOUTH_COUNTER >= MOUTH_ASPECT_RATIO_CONSEC_FRAMES:
                        if AUDIO_ENABLED:
                            pygame.mixer.music.play(-1)
                        current_status_message = "You are Drowsy (Yawning)"
                        drowsy_detected = True
                        break  # Exit for loop once drowsiness detected
                else:
                    MOUTH_COUNTER = 0

            if not drowsy_detected:
                current_status_message = "Normal"

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        if vs is None:
            break

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera_route():
    start_camera()
    return 'Camera started successfully!'

@app.route('/stop_camera', methods=['POST'])
def stop_camera_route():
    stop_camera()
    return 'Camera stopped successfully!'

@app.route('/status_message')
def get_status_message():
    global current_status_message
    return jsonify(status=current_status_message)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global EYE_ASPECT_RATIO_THRESHOLD, EYE_ASPECT_RATIO_CONSEC_FRAMES
    global MOUTH_ASPECT_RATIO_THRESHOLD, MOUTH_ASPECT_RATIO_CONSEC_FRAMES
    global AUDIO_ENABLED

    data = request.get_json()
    EYE_ASPECT_RATIO_THRESHOLD = data.get('eyeAspectRatioThreshold', EYE_ASPECT_RATIO_THRESHOLD)
    EYE_ASPECT_RATIO_CONSEC_FRAMES = data.get('eyeAspectRatioFrames', EYE_ASPECT_RATIO_CONSEC_FRAMES)
    MOUTH_ASPECT_RATIO_THRESHOLD = data.get('mouthAspectRatioThreshold', MOUTH_ASPECT_RATIO_THRESHOLD)
    MOUTH_ASPECT_RATIO_CONSEC_FRAMES = data.get('mouthAspectRatioFrames', MOUTH_ASPECT_RATIO_CONSEC_FRAMES)
    AUDIO_ENABLED = data.get('audioEnabled', AUDIO_ENABLED)

    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
