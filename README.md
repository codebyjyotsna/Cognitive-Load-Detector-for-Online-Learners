# Cognitive-Load-Detector-for-Online-Learners
This project aims to help online students improve their learning efficiency by detecting cognitive load using webcam data (eye movement, blink rate, etc.). The system predicts cognitive load and provides timely alerts to the learner, ensuring they stay focused and engaged.

## Features
- **Eye Tracking**: Detect eye movement and measure the Eye Aspect Ratio (EAR) using OpenCV and Dlib.
- **Blink Detection**: Identify blinks based on EAR thresholds.
- **Cognitive Load Prediction**: Use machine learning models (KNN/Random Forest) to predict cognitive load based on webcam data.
- **Real-Time Feedback**: Provide visual and voice feedback to learners.
- **Platform Integration**: Integrate with online learning platforms like Coursera and Moodle.
- **Data Visualization**: Historical visualization for learners to recognize focus patterns.

## Tech Stack
- **OpenCV**: For blink detection and image processing.
- **Dlib**: For facial landmark detection and eye tracking.
- **Machine Learning**: KNN and Random Forest for cognitive load prediction.
- **PyGame**: For building an interactive user interface.
- **Flask/Django** (optional): For API integration with learning platforms.
