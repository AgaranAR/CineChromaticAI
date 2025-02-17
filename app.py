import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import script
import emotion
import cv2
import numpy as np
import pdfplumber
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'mp4', 'mov', 'avi'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'script' not in request.files or 'video' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        script_file = request.files['script']
        video_file = request.files['video']

        if not (script_file and video_file):
            return jsonify({'error': 'No file selected'}), 400

        if not (allowed_file(script_file.filename) and allowed_file(video_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save files
        script_path = os.path.join(UPLOAD_FOLDER, secure_filename(script_file.filename))
        video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))

        script_file.save(script_path)
        video_file.save(video_path)

        # Process script
        if script_path.endswith('.pdf'):
            script_text = extract_text_from_pdf(script_path)
        else:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_text = f.read()

        if not script_text:
            return jsonify({'error': 'Could not extract text from script'}), 400

        # Get color information
        color_info = script.get_dominant_color(script_text)
        logger.debug(f"Color info: {color_info}")

        # Process video
        output_path = os.path.join(UPLOAD_FOLDER, f"processed_{secure_filename(video_file.filename)}")
        apply_color_grading(video_path, output_path, color_info['color'])

        return jsonify({
            'dominant_emotion': color_info['emotion'],
            'color': color_info['color'],
            'output_video': output_path
        })

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download(filename):
    try:
        full_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(full_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

def apply_color_grading(input_path, output_path, color_hex):
    """Apply color grading to video frames."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Convert hex color to BGR
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect emotion from frame
            frame_emotion = emotion.detect_emotion(frame)

            # Apply emotion-based filter
            frame = emotion.apply_emotion_based_filter(frame, frame_emotion)

            # Apply overall color grading
            frame = cv2.addWeighted(
                frame, 0.7,
                np.full(frame.shape, color, dtype=np.uint8), 0.3,
                0
            )

            out.write(frame)

        cap.release()
        out.release()

    except Exception as e:
        logger.error(f"Color grading error: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)