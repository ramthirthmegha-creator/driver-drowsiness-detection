import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import threading

# Load the trained model
model = load_model("drowsiness_model.h5")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize pygame mixer for sound
pygame.mixer.init()

# Define the alarm sound function
def play_alarm():
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()

# Define class labels
labels = ['Closed', 'Open']

# Face and eye detection using Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Start video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255.0
            eye = np.expand_dims(eye, axis=0)

            prediction = model.predict(eye)
            label = labels[int(prediction[0][0] > 0.5)]

            if label == "Closed":
                cv2.putText(frame, "DROWSY 😴", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                threading.Thread(target=play_alarm).start()
            else:
                cv2.putText(frame, "AWAKE 😀", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            break  # Analyze one eye per face to avoid overlap

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()