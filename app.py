from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

import os

model_path = os.path.join(os.path.dirname(__file__), 'emotion_detection_model.h5')
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

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 0, 0), -1)
                cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global camera
    if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
        camera.release()
        camera = None
    return '', 204

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
