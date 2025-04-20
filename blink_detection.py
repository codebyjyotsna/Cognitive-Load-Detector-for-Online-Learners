from src.core.eye_tracking import get_eye_aspect_ratio, detect_eyes

def detect_blinks(frame, threshold=0.2):
    """
    Detect blinks based on the Eye Aspect Ratio (EAR) threshold.
    """
    left_eye, right_eye = detect_eyes(frame)
    if left_eye and right_eye:
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        if ear < threshold:
            return True  # Blink detected
    return False
