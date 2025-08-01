# Real-Time Emotion Detection using CNN, OpenCV, and Flask
This project is a complete real-time emotion detection system. It uses a custom-trained Convolutional Neural Network (CNN) to identify emotions from a live webcam feed, which is served through a Python Flask web application.

## 🎥 Demo
(Note: This is a placeholder GIF. You can replace it with a screen recording of your actual application.)

## ✨ Features
- Real-Time Video Streaming: Captures video from your webcam and streams it to a web interface using Flask and OpenCV.

- Accurate Face Detection: Utilizes OpenCV's Haar Cascade classifier to robustly detect faces in the video feed.

- Deep Learning for Emotion Recognition: Employs a custom Keras/TensorFlow CNN model trained on the FER2013 dataset to classify faces into one of seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.

- Complete Project Documentation: Includes the Jupyter Notebook used for model training, providing full transparency into the development process.

- Standalone Executable: Comes with a pre-compiled .exe file for easy execution on Windows without any setup.

- Simple Web Interface: A clean HTML page to view the live, annotated video feed.

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
└── templates/
    └── styles.css                 # CSS Styling
    └── camera.jpg                 # jpg 
```
## ⚙️ How It Works
The application workflow is as follows:

- **Flask Backend:** The app.py script initializes a Flask web server to handle requests.

- **Video Capture:** When the user navigates to the homepage, the browser requests the video feed from the /video endpoint. OpenCV then accesses the default webcam to start capturing frames.

- **Face Detection:** Each captured frame is converted to grayscale. The pre-trained Haar Cascade classifier (haarcascade_frontalface_default.xml) scans the frame to identify the coordinates of any faces.

- **Emotion Prediction:**

  - For each detected face, the region of interest is cropped and resized to 48x48 pixels to match the model's input size.

  - This grayscale face image is normalized and fed into the pre-trained Keras model (emotion_detection_model.h5).

  - The model predicts the emotion, outputting a probability for each of the seven classes. The class with the highest probability is chosen as the final prediction.

  - Annotation & Streaming: A blue rectangle is drawn around the detected face on the original video frame, and the predicted emotion is written as text above it. This annotated frame is then encoded as a JPEG and streamed to the browser.

## 🧠 Model Details
The emotion recognition model is a Convolutional Neural Network (CNN) built with Keras and trained on the FER2013 dataset.

### Data Processing
Input images are 48x48 grayscale pictures of faces. To improve model robustness, the training data was augmented with random rotations, shears, zooms, and horizontal flips. All image pixel values were normalized to a [0, 1] range.
### CNN Architecture
The model is a sequential CNN designed for image classification. It consists of four main convolutional blocks, each containing Conv2D, ReLU activation, MaxPooling2D, and Dropout layers. This structure progressively extracts complex features while reducing dimensionality and preventing overfitting.
### Training
The model was trained for 30 epochs, and the final weights were saved to the emotion_detection_model.h5 file.
## 🛠️ Setup and Installation
Choose one of the two methods below to run the application.

### Method 1: Running the Executable (Easiest)
- No setup required! This is for users who just want to try out the application on a Windows machine.

- Ensure all the files (emotion_detector.exe, model_file_30epochs.h5, haarcascade_frontalface_default.xml, and the templates folder) are in the same directory.

- Double-click emotion_detector.exe. A command prompt window will appear, indicating the server is running.

- Open your web browser and navigate to `http://127.0.0.1:5000`.

### Method 2: Running from Python (For Developers)
Follow these steps to run the application from the source code.

**Prerequisites**
- Python 3.7+
- A webcam connected to your computer.

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
Running on http://127.0.0.1:5000
```
Open your web browser and navigate to `http://127.0.0.1:5000`.

You should now see the live feed from your webcam with emotion detection running!

## 🛑 Stopping the Application
To stop the Flask server, go to the terminal or command prompt window where it is running and press `Ctrl+C`.

