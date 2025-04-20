import pygame
from src.core.blink_detection import detect_blinks
from src.core.cognitive_load import predict_cognitive_load

pygame.init()

# UI settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Cognitive Load Detector")

def run_ui(webcam_feed):
    """
    Main UI loop to display cognitive load and alerts.
    """
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
            alert_message = "Blink Detected! Stay Focused."
            print(alert_message)
        
        # Predict cognitive load
        features = [/* populate features like blink rate, etc. */]
        load_prediction = predict_cognitive_load(features, model_type="knn")
        
        # Display prediction on UI
        screen.fill((255, 255, 255))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Cognitive Load: {load_prediction}", True, (0, 0, 0))
        screen.blit(text, (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 3))
        
        pygame.display.flip()
