from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

app = Flask(__name__)

# Load face detector and mask detector
prototxt_path = "face_detector/deploy.prototxt"
weights_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
try:
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    mask_net = load_model("mask_detector.h5")
    print("[INFO] Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Function to detect and classify masks
def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = mask_net.predict(face, verbose=0)[0]
            if len(prediction) == 1:
                mask_prob = prediction[0]
                label = "Mask" if mask_prob > 0.7 else "No Mask"
                print(f"[DEBUG] Sigmoid Prediction: {mask_prob}, Label: {label}")
            else:
                no_mask_prob, mask_prob = prediction[0], prediction[1]
                label = "Mask" if mask_prob > no_mask_prob else "No Mask"
                print(f"[DEBUG] Softmax Prediction: No Mask={no_mask_prob}, Mask={mask_prob}, Label: {label}")

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label} ({mask_prob:.2f})", (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            print(f"[DEBUG] Frame dimensions after processing: {frame.shape}")

    return frame

# Video feed generator
def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = detect_mask(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("[INFO] Starting video feed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        file_path = os.path.join("static", "uploaded_image.jpg")
        file.save(file_path)
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Could not load uploaded image.")
            return render_template('index.html', error="Invalid image file.")
        print(f"[DEBUG] Original image dimensions: {image.shape}")

        processed_image = detect_mask(image)
        cv2.imwrite(file_path, processed_image)
        print("[INFO] Image processed and saved")
        timestamp = int(time.time())
        return render_template('index.html', uploaded_image="uploaded_image.jpg", timestamp=timestamp)
    return render_template('index.html', error="No file uploaded.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)