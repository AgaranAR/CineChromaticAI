import spacy
import logging
import google.generativeai as genai
import json
import re
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    nlp = None

# Predefined emotion-to-color mapping with expanded emotions
COLOR_MAPPING = {
    # Dark/Mysterious tones
    "dark": "#1B263B",
    "mystery": "#1B263B",
    "horror": "#000000",
    "suspense": "#2F4F4F",

    # Warm/Emotional tones
    "love": "#D72638",
    "romance": "#D72638",
    "passion": "#FF1493",
    "anger": "#FF4500",

    # Light/Positive tones
    "hope": "#FFD700",
    "joy": "#FFA500",
    "happiness": "#FFD700",

    # Fantasy/Magical tones
    "fantasy": "#8A2BE2",
    "magic": "#9400D3",
    "supernatural": "#9370DB",

    # Action/Adventure tones
    "action": "#FF4500",
    "adventure": "#FF8C00",
    "danger": "#FF4500",

    # Sci-fi/Tech tones
    "sci-fi": "#32CD32",
    "futuristic": "#00FF7F",
    "technological": "#4169E1",

    # Crime/Noir tones
    "crime": "#2F4F4F",
    "noir": "#1A1A1A",
    "detective": "#4682B4",
}

def analyze_script_with_gemini(script_text):
    """
    Analyzes script using Gemini AI for emotion detection.
    """
    try:
        prompt = f"""
        Analyze this movie script excerpt and determine the dominant emotion and tone.
        Return a JSON object with these fields:
        - dominant_emotion: The main emotional tone
        - secondary_emotions: List of supporting emotional tones
        - intensity: Rating from 1-10 of emotional intensity

        Script excerpt:
        {script_text[:2000]}
        """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(0))
            return analysis
        return None

    except Exception as e:
        logger.error(f"Gemini AI analysis error: {str(e)}")
        return None

def extract_emotions(script_text):
    """
    Extracts emotion-related keywords from the script using spaCy.
    Falls back to basic keyword matching if spaCy fails.
    """
    try:
        if not nlp:
            # Fallback to basic keyword matching
            emotions = [emotion for emotion in COLOR_MAPPING.keys() 
                       if emotion in script_text.lower()]
            return emotions[:3]  # Return top 3 matches

        doc = nlp(script_text.lower())
        emotion_counts = defaultdict(int)

        # Count emotion keywords
        for token in doc:
            if token.lemma_ in COLOR_MAPPING:
                emotion_counts[token.lemma_] += 1

        # Get emotions sorted by frequency
        sorted_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [emotion for emotion, _ in sorted_emotions]

    except Exception as e:
        logger.error(f"Error extracting emotions: {str(e)}")
        return []

def get_dominant_color(script_text):
    """
    Determines the dominant color based on script analysis.
    Uses both spaCy and Gemini AI for robust emotion detection.
    """
    try:
        # First try spaCy analysis
        emotions = extract_emotions(script_text)

        # If spaCy found emotions, use the most frequent one
        if emotions:
            primary_emotion = emotions[0]
            return {
                "color": COLOR_MAPPING.get(primary_emotion, "#FFFFFF"),
                "emotion": primary_emotion
            }

        # Fallback to Gemini AI analysis
        gemini_analysis = analyze_script_with_gemini(script_text)
        if gemini_analysis and 'dominant_emotion' in gemini_analysis:
            emotion = gemini_analysis['dominant_emotion'].lower()
            return {
                "color": COLOR_MAPPING.get(emotion, "#FFFFFF"),
                "emotion": emotion
            }

        # Default fallback
        return {
            "color": "#FFFFFF",
            "emotion": "neutral"
        }

    except Exception as e:
        logger.error(f"Error getting dominant color: {str(e)}")
        return {
            "color": "#FFFFFF",
            "emotion": "neutral"
        }