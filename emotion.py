"""
emotion.py — optional face-emotion sampler

This module is NO LONGER part of the core color-grading pipeline.
The canonical emotion driving all visual grading comes from script.py
(the script text), not from individual video frames.

This file is kept as a standalone utility you can call separately
if you want to compare script intent vs on-screen performance.
It does NOT affect the output video.

Usage (standalone, not imported by app.py):
    from emotion import sample_video_emotion
    emotion = sample_video_emotion("my_clip.mp4")
"""

import cv2
import logging
from collections import Counter, deque

logger = logging.getLogger(__name__)

# Only import DeepFace when this module is actually used
_deepface = None

def _get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
        except ImportError:
            logger.error("DeepFace is not installed. Run: pip install deepface")
            raise
    return _deepface


def sample_video_emotion(video_path: str, sample_every_n_frames: int = 30) -> str:
    """
    Sample a video at regular intervals and return the most common detected emotion.

    Args:
        video_path: path to the video file
        sample_every_n_frames: how often to sample (default: every 30 frames = ~1s at 30fps)

    Returns:
        One of: "happy" | "sad" | "angry" | "fear" | "surprise" | "disgust" | "neutral"
    """
    DeepFace = _get_deepface()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return "neutral"

    history = deque(maxlen=50)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n_frames != 0:
            continue

        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(results, list) and results:
                emotions = results[0].get("emotion", {})
                if emotions:
                    history.append(max(emotions, key=emotions.get))
        except Exception as e:
            logger.debug(f"Frame {frame_idx} detection skipped: {e}")

    cap.release()

    if not history:
        return "neutral"

    dominant = Counter(history).most_common(1)[0][0]
    logger.debug(f"Video emotion sample result: {dominant} (from {len(history)} samples)")
    return dominant
