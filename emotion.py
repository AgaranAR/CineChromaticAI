import cv2
import logging
from collections import Counter, deque
import numpy as np
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Emotion history for smoothing
emotion_history = deque(maxlen=30)  
frame_count = 0  # Track frames to reduce DeepFace calls

# Emotion to color mapping (BGR format)
EMOTION_COLORS = {
    'happy': (0, 215, 255),    # Gold
    'sad': (255, 0, 0),        # Blue
    'angry': (0, 0, 255),      # Red
    'fear': (128, 0, 128),     # Purple
    'surprise': (0, 165, 255), # Orange
    'neutral': (128, 128, 128), # Gray
    'disgust': (75, 0, 130),   # Indigo
}

def preprocess_frame(frame):
    """Preprocess frame for emotion detection."""
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        resized = cv2.resize(frame, (224, 224))  
        return resized.astype(np.float32) / 255.0  
    except Exception as e:
        logger.error(f"Error preprocessing frame: {str(e)}")
        return None

def detect_emotion(frame):
    """
    Detect emotion in a frame using DeepFace.
    Processes every 5th frame for efficiency.
    """
    global frame_count
    frame_count += 1

    if frame_count % 5 != 0:  
        if emotion_history:
            return Counter(emotion_history).most_common(1)[0][0]  
        return 'neutral'

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(results, list) and results:
            emotions = results[0].get('emotion', {})
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)  
                emotion_history.append(dominant_emotion)
                smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
                return smoothed_emotion

        return 'neutral'

    except Exception as e:
        logger.error(f"Emotion detection error: {str(e)}")
        return 'neutral'

def apply_emotion_based_filter(frame, emotion, confidence=0.7):
    """
    Applies a visual filter based on detected emotion.
    Overlay intensity is based on confidence level.
    """
    try:
        color_bgr = EMOTION_COLORS.get(emotion, (128, 128, 128))  
        overlay = np.full(frame.shape, color_bgr, dtype=np.uint8)

        # Adjust intensity dynamically based on confidence level (higher confidence = stronger effect)
        alpha = min(0.4, max(0.1, confidence))  
        
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    except Exception as e:
        logger.error(f"Error applying filter: {str(e)}")
        return frame  
