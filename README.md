#Emotion Detector 
## Real-Time Emotion Detection using CNN, OpenCV, and Flask
This project is a complete real-time emotion detection system. It uses a custom-trained Convolutional Neural Network (CNN) to identify emotions from a live webcam feed, which is served through a Python Flask web application.

## 🎥 Demo
(Note: This is a placeholder GIF. You can replace it with a screen recording of your actual application.)

## ✨ Features
* Real-Time Video Streaming: Captures video from your webcam and streams it to a web interface using Flask and OpenCV.
* Accurate Face Detection: Utilizes OpenCV's Haar Cascade classifier to robustly detect faces in the video feed.
* Deep Learning for Emotion Recognition: Employs a custom Keras/TensorFlow CNN model trained on the FER2013 dataset to classify faces into one of seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.
* Complete Project Documentation: Includes the Jupyter Notebook used for model training, providing full transparency into the development process.
* Standalone Executable: Comes with a pre-compiled .exe file for easy execution on Windows without any setup.
* Simple Web Interface: A clean HTML page to view the live, annotated video feed.

## 📂 Project Structure
Your project directory should be set up as follows for the application to run correctly:
```
.
├── Emotion_Detection_Model.ipynb  # Jupyter notebook for model training
├── model_file_30epochs.h5         # The pre-trained Keras model
├── haarcascade_frontalface_default.xml # OpenCV face detection classifier
├── app.py                         # The main Flask application
├── requirements.txt               # Python dependencies
├── emotion_detector.exe           # Standalone executable for Windows
└── templates/
    └── index.html                 # Frontend web page
```
## 🧠 Model Training
The deep learning model used for emotion recognition was trained from scratch using a Convolutional Neural Network (CNN).

Notebook: The entire training process is documented in the Emotion_Detection_Model.ipynb Jupyter Notebook.

Dataset: The model was trained on the well-known FER2013 (Facial Expression Recognition 2013) dataset from a Kaggle competition. This dataset contains tens of thousands of 48x48 pixel grayscale images of faces, each labeled with one of the seven core emotions.

Architecture: The notebook details the CNN architecture, data preprocessing steps, and training loops used to build the model_file_30epochs.h5 model file.

## ⚙️ How It Works
Flask Backend: The app.py script initializes a Flask web server.

Video Capture: When the main page is loaded, it requests the video feed from the /video endpoint. OpenCV captures frames from the default webcam.

Face Detection: Each frame is converted to grayscale. The Haar Cascade classifier (haarcascade_frontalface_default.xml) identifies the coordinates of any faces.

Emotion Prediction:

For each detected face, the region of interest is extracted and resized to 48x48 pixels.

This image is normalized and reshaped to match the input requirements of the CNN model.

The pre-trained model (model_file_30epochs.h5) predicts the emotion.

The emotion with the highest probability is chosen as the result.

Annotation & Streaming: A blue rectangle is drawn around the detected face, and the predicted emotion is written above it. The final frame is encoded as a JPEG and streamed to the browser.

## 🛠️ Setup and Installation
Choose one of the two methods below to run the application.

### Method 1: Running the Executable (Easiest)
No setup required! This is for users who just want to try out the application on a Windows machine.

Ensure all the files (emotion_detector.exe, model_file_30epochs.h5, haarcascade_frontalface_default.xml, and the templates folder) are in the same directory.

Double-click emotion_detector.exe. A command prompt window will appear, indicating the server is running.

Open your web browser and navigate to https://www.google.com/search?q=http://127.0.0.1:5000.

### Method 2: Running from Python (For Developers)
Follow these steps to run the application from the source code.

**Prerequisites**
* Python 3.7+
* A webcam connected to your computer.

**1. Install Dependencies**
Open your terminal or command prompt, navigate to your project directory, and run the following command to install the required Python libraries from the requirements.txt file:
```
pip install -r requirements.txt
```
**2. Run the Application**
Make sure you are in the project's root directory in your terminal.

Run the Flask application with the following command:
```
python app.py
```
You should see output indicating that the Flask server is running:
```
 * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
```
Open your web browser and navigate to `http://127.0.0.1:5000`.

You should now see the live webpage!

## 🛑 Stopping the Application
To stop the Flask server, go to the terminal or command prompt window where it is running and press Ctrl+C.
