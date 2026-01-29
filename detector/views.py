from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
import cv2
import numpy as np
from keras.models import load_model
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.py')  # adjust to your model filename
model = load_model(model_path)
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
camera = None

def gen_frames():
    global camera
    if camera is None or not (hasattr(camera, 'isOpened') and camera.isOpened()):
        camera = cv2.VideoCapture(0)
    while True:
        if camera is None:
            break
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                sub_face = gray[y:y+h, x:x+w]
                resized = cv2.resize(sub_face, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                result = model.predict(reshaped, verbose=0)
                label = np.argmax(result, axis=1)[0]
                emotion = labels_dict[label]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 0, 0), -1)
                cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def index(request):
    return render(request, 'index.html')

def video(request):
    return StreamingHttpResponse(gen_frames(),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def stop(request):
    global camera
    if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
        camera.release()
        camera = None
    return HttpResponse(status=204)