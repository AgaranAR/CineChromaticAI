import cv2
import logging
from collections import Counter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store detected emotions across frames for smoothing
emotion_history = []
MAX_HISTORY = 30  # Keep track of last 30 frames

# Emotion to color mapping for visual feedback
EMOTION_COLORS = {
    'happy': '#FFD700',  # Gold
    'sad': '#0000FF',    # Blue
    'angry': '#FF0000',  # Red
    'fear': '#800080',   # Purple
    'surprise': '#FFA500',  # Orange
    'neutral': '#808080',   # Gray
    'disgust': '#4B0082',
}

def preprocess_frame(frame):
    """Preprocess frame for emotion detection."""
    try:
        # Resize for faster processing
        frame = cv2.resize(frame, (224, 224))
        return frame
    except Exception as e:
        logger.error(f"Error preprocessing frame: {str(e)}")
        return None

def detect_emotion(frame):
    """
    Detects emotion in a frame. Falls back to neutral if detection fails.
    Returns the most stable emotion based on recent history.
    """
    try:
        # For now, return a default emotion since we're having TensorFlow issues
        # This will be replaced with actual detection once dependencies are fixed
        emotion = "neutral"

        # Update emotion history
        emotion_history.append(emotion)
        if len(emotion_history) > MAX_HISTORY:
            emotion_history.pop(0)

        # Get most stable emotion from history
        if emotion_history:
            stable_emotion = Counter(emotion_history).most_common(1)[0][0]
        else:
            stable_emotion = "neutral"

        logger.debug(f"Detected emotion: {stable_emotion}")
        return stable_emotion

    except Exception as e:
        logger.error(f"Error detecting emotion: {str(e)}")
        return "neutral"

def get_emotion_color(emotion):
    """Returns the corresponding color for an emotion."""
    return EMOTION_COLORS.get(emotion, '#808080')

def apply_emotion_based_filter(frame, emotion):
    """
    Applies visual filters based on detected emotion.
    """
    try:
        # Convert emotion color to BGR
        color_hex = get_emotion_color(emotion)
        color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Create color overlay
        overlay = np.full(frame.shape, color_bgr, dtype=np.uint8)

        # Blend based on emotion intensity
        alpha = 0.2  # Adjust overlay intensity
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    except Exception as e:
        logger.error(f"Error applying emotion filter: {str(e)}")
        return frame