import cv2
from src.ui.main_ui import run_ui

def main():
    webcam_feed = cv2.VideoCapture(0)
    run_ui(webcam_feed)
    webcam_feed.release()

if __name__ == "__main__":
    main()
