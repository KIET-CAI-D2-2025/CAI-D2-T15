import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained model
model = load_model("mask_detector.h5")

# Load image
image_path = r"C:\Users\Sridh\Desktop\Face-Mask-Detection\dataset\with_mask\0-with-mask.jpg"  # Replace with your image path
frame = cv2.imread(image_path)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

for (x, y, w, h) in faces:
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    (mask, withoutMask) = model.predict(face)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

cv2.imwrite("output_image.jpg", frame)
cv2.imshow("Face Mask Detector", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()