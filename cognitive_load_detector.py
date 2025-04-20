import cv2
import dlib
import pygame
import pyttsx3
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Dlib and OpenCV
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/models/shape_predictor_68_face_landmarks.dat")

# Load pre-trained ML models
knn_model = pickle.load(open("data/models/knn_model.pkl", "rb"))
random_forest_model = pickle.load(open("data/models/random_forest.pkl", "rb"))

# Initialize PyGame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Cognitive Load Detector")

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Historical Data Storage
data = pd.DataFrame(columns=["timestamp", "cognitive_load"])

# Configuration
THRESHOLD_BLINK = 0.2
THRESHOLD_COG_LOAD = 0.5


# Core Eye Tracking Functionality
def get_eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect eye activity.
    """
    p1 = ((eye_points[1][0] - eye_points[0][0]) ** 2 + (eye_points[1][1] - eye_points[0][1]) ** 2) ** 0.5
    p2 = ((eye_points[5][0] - eye_points[4][0]) ** 2 + (eye_points[5][1] - eye_points[4][1]) ** 2) ** 0.5
    p3 = ((eye_points[3][0] - eye_points[2][0]) ** 2 + (eye_points[3][1] - eye_points[2][1]) ** 2) ** 0.5

    ear = (p1 + p2) / (2.0 * p3)
    return ear


def detect_eyes(frame):
    """
    Detect eyes and facial landmarks using dlib.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        # Extract eye points (left and right)
        for n in range(36, 42):  # Left eye
            left_eye.append((landmarks.part(n).x, landmarks.part(n).y))
        for n in range(42, 48):  # Right eye
            right_eye.append((landmarks.part(n).x, landmarks.part(n).y))

        return left_eye, right_eye
    return None, None


def detect_blinks(frame):
    """
    Detect blinks based on the Eye Aspect Ratio (EAR) threshold.
    """
    left_eye, right_eye = detect_eyes(frame)
    if left_eye and right_eye:
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        if ear < THRESHOLD_BLINK:
            return True  # Blink detected
    return False


# Cognitive Load Prediction
def predict_cognitive_load(features, model_type="knn"):
    """
    Predict cognitive load by using either KNN or Random Forest models.
    """
    features = np.array(features).reshape(1, -1)
    if model_type == "knn":
        return knn_model.predict(features)[0]
    elif model_type == "random_forest":
        return random_forest_model.predict(features)[0]


# Voice Alerts
def voice_alert(message):
    """
    Provide a voice alert using Text-to-Speech.
    """
    engine.say(message)
    engine.runAndWait()


# Historical Data Functions
def store_data(cognitive_load):
    """
    Store cognitive load data for historical visualization.
    """
    global data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"timestamp": timestamp, "cognitive_load": cognitive_load}
    data = data.append(new_entry, ignore_index=True)


def plot_cognitive_load():
    """
    Plot cognitive load over time for visualization.
    """
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    plt.plot(data["timestamp"], data["cognitive_load"], label="Cognitive Load")
    plt.xlabel("Time")
    plt.ylabel("Cognitive Load")
    plt.title("Cognitive Load Over Time")
    plt.legend()
    plt.show()


# Gamification
points = 0


def reward_points(focus_duration):
    """
    Reward points based on focus duration.
    """
    global points
    if focus_duration >= 30:  # Focus duration in minutes
        points += 10
        print("You earned 10 points!")
    return points


# Main UI Functionality
def run_ui():
    """
    Main UI loop to display cognitive load and alerts.
    """
    global data
    webcam_feed = cv2.VideoCapture(0)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = webcam_feed.read()
        if not ret:
            break

        # Detect blinks
        blink_detected = detect_blinks(frame)
        if blink_detected:
            voice_alert("Blink detected! Stay focused.")

        # Predict cognitive load
        features = [/* Populate features like blink rate, etc. */]
        load_prediction = predict_cognitive_load(features, model_type="knn")
        store_data(load_prediction)

        if load_prediction > THRESHOLD_COG_LOAD:
            voice_alert("Your cognitive load is high. Take a break!")

        # Display prediction on UI
        screen.fill((255, 255, 255))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Cognitive Load: {load_prediction}", True, (0, 0, 0))
        screen.blit(text, (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 3))

        pygame.display.flip()

    webcam_feed.release()
    pygame.quit()


# Run the Application
if __name__ == "__main__":
    run_ui()
    plot_cognitive_load()
