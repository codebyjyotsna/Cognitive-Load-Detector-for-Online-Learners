import cv2
import dlib

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/models/shape_predictor_68_face_landmarks.dat")

def get_eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect eye activity.
    EAR is used to measure eye openness and blinking.
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
