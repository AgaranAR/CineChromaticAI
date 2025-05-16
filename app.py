import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import script  # Assuming this exists
from emotion import detect_emotion  # Assuming this exists
import cv2
import numpy as np
import pdfplumber
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'mp4', 'mov', 'avi'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini AI
GENAI_API_KEY = "AIzaSyA24X_DTrvDKLOPNoh2dYY1uV_xtGvhTvc"  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)

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

        # Save files with proper error handling
        try:
            script_path = os.path.join(UPLOAD_FOLDER, secure_filename(script_file.filename))
            video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))

            script_file.save(script_path)
            video_file.save(video_path)

            logger.debug(f"Files saved: Script={script_path}, Video={video_path}")
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            return jsonify({'error': 'Failed to save uploaded files'}), 500

        # Process script
        try:
            if script_path.endswith('.pdf'):
                script_text = extract_text_from_pdf(script_path)
            else:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_text = f.read()

            if not script_text:
                return jsonify({'error': 'Could not extract text from script'}), 400
        except Exception as e:
            logger.error(f"Script processing error: {str(e)}")
            return jsonify({'error': 'Failed to process script file'}), 500

        # Get color information
        try:
          color_info = script.get_dominant_color(script_text)
          logger.debug(f"Color info: {color_info}")
          logger.debug(f"Type of color_info['color']: {type(color_info['color'])}") #added line
        except Exception as e:
            logger.error(f"Color analysis error: {str(e)}")
            return jsonify({'error': 'Failed to analyze script emotions'}), 500

        # Process video
        try:
            output_filename = f"processed_{os.path.splitext(secure_filename(video_file.filename))[0]}.mp4"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)

            logger.debug(f"Processing video: Input={video_path}, Output={output_path}")
            result = apply_cinema_grade_color(video_path, output_path, color_info['color'])
            
            # Check if the result is a dictionary (new enhanced function) or a string (legacy function)
            if isinstance(result, dict):
                output_path = result['output_path']
                scene_info = {
                    'detected_scenes': result['detected_scenes'],
                    'scene_start_frames': result['scene_start_frames']
                }
            else:
                output_path = result
                scene_info = {'detected_scenes': 1, 'scene_start_frames': [0]}

            if not os.path.exists(output_path):
                logger.error(f"Output video file not found at {output_path}")
                return jsonify({'error': 'Video processing failed'}), 500

            logger.debug(f"Output video successfully created at {output_path}.")

            #Delete original files.
            try:
                os.remove(script_path)
                os.remove(video_path)
            except Exception as e:
                logger.warning(f"Failed to delete original files: {e}")

            return jsonify({
                'dominant_emotion': color_info['emotion'],
                'color': color_info['color'],
                'output_video': output_filename,
                'scene_info': scene_info
            })

        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return jsonify({'error': f'Failed to process video: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    logger.debug(f"Download requested for file: {file_path}")

    # Wait a few seconds if the file doesn't exist yet
    for _ in range(5):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return send_file(file_path, as_attachment=True)
        logger.warning(f"File not ready yet, retrying... ({file_path})")
        time.sleep(1)

    logger.error(f"File {file_path} not found after retries.")
    return jsonify({'error': 'File not found'}), 404

def convert_hex_to_bgr(color_hex):
    """Converts hex color to BGR with gamma correction or handles BGR tuples."""
    if isinstance(color_hex, str):
        try:
            color_bgr = tuple(int(((int(color_hex.lstrip('#')[i:i+2], 16) / 255) ** 2.2) * 255) for i in (4, 2, 0))
            return color_bgr
        except (ValueError, IndexError):
            logger.warning(f"Invalid color hex: {color_hex}. Using default neutral color.")
            return (128, 128, 128)
    elif isinstance(color_hex, tuple) and len(color_hex) == 3:
        return color_hex # return the tuple directly.
    else:
        logger.warning(f"Invalid color format: {color_hex}. Using default neutral color.")
        return (128, 128, 128)

def enhance_color_grading_functions():
    """Return the emotion profiles with enhanced parameters"""
    return {
        'happy': {
            'contrast': 1.2,
            'saturation': 1.15,
            'shadow_lift': 1.1,
            'highlight_roll': 0.8,
            'color_intensity': 0.4,
            'halation_strength': 0.25,
            'grain_intensity': 0.015,
            'color_temp': 1.1,
            'tint': 0.95,
            # New parameters
            'white_level': 245,
            'black_level': 16,
            'midtone_contrast': 1.1,
            'shadow_tint': (1.05, 1.0, 0.95),  # R,G,B tint for shadows (warmer shadows)
            'highlight_tint': (1.05, 1.02, 0.9),  # R,G,B tint for highlights (golden highlights)
            'vignette_amount': 0.1,
            'sharpness': 0.2,
            'noise_reduction': 0.1,
        },
        'sad': {
            'contrast': 0.95,
            'saturation': 0.85,
            'shadow_lift': 1.2,
            'highlight_roll': 0.7,
            'color_intensity': 0.5,
            'halation_strength': 0.15,
            'grain_intensity': 0.02,
            'color_temp': 0.9,
            'tint': 1.05,
            # New parameters
            'white_level': 235,
            'black_level': 20,
            'midtone_contrast': 0.9,
            'shadow_tint': (0.9, 0.95, 1.1),  # R,G,B tint for shadows (cool shadows)
            'highlight_tint': (0.95, 0.95, 1.05),  # R,G,B tint for highlights (cool highlights)
            'vignette_amount': 0.25,
            'sharpness': 0.1,
            'noise_reduction': 0.2,
        },
        'angry': {
            'contrast': 1.3,
            'saturation': 0.9,
            'shadow_lift': 0.9,
            'highlight_roll': 0.65,
            'color_intensity': 0.6,
            'halation_strength': 0.1,
            'grain_intensity': 0.025,
            'color_temp': 1.15,
            'tint': 0.9,
            # New parameters
            'white_level': 250,
            'black_level': 25,
            'midtone_contrast': 1.2,
            'shadow_tint': (1.1, 0.9, 0.9),  # R,G,B tint for shadows (red shadows)
            'highlight_tint': (1.05, 0.95, 0.9),  # R,G,B tint for highlights (reddish highlights)
            'vignette_amount': 0.3,
            'sharpness': 0.3,
            'noise_reduction': 0.05,
        },
        'fearful': {
            'contrast': 1.1,
            'saturation': 0.8,
            'shadow_lift': 1.3,
            'highlight_roll': 0.6,
            'color_intensity': 0.45,
            'halation_strength': 0.2,
            'grain_intensity': 0.022,
            'color_temp': 0.85,
            'tint': 1.1,
            # New parameters
            'white_level': 230,
            'black_level': 20,
            'midtone_contrast': 1.0,
            'shadow_tint': (0.85, 0.9, 1.15),  # R,G,B tint for shadows (blue shadows)
            'highlight_tint': (0.9, 0.9, 1.05),  # R,G,B tint for highlights (cool highlights)
            'vignette_amount': 0.35,
            'sharpness': 0.15,
            'noise_reduction': 0.1,
        },
        'surprised': {
            'contrast': 1.15,
            'saturation': 1.1,
            'shadow_lift': 1.0,
            'highlight_roll': 0.75,
            'color_intensity': 0.35,
            'halation_strength': 0.3,
            'grain_intensity': 0.018,
            'color_temp': 1.0,
            'tint': 1.0,
            # New parameters
            'white_level': 245,
            'black_level': 15,
            'midtone_contrast': 1.05,
            'shadow_tint': (0.95, 1.0, 1.05),  # R,G,B tint for shadows (slightly cool shadows)
            'highlight_tint': (1.02, 1.02, 1.0),  # R,G,B tint for highlights (bright highlights)
            'vignette_amount': 0.15,
            'sharpness': 0.25,
            'noise_reduction': 0.15,
        },
        'neutral': {
            'contrast': 1.05,
            'saturation': 1.0,
            'shadow_lift': 1.0,
            'highlight_roll': 0.75,
            'color_intensity': 0.3,
            'halation_strength': 0.2,
            'grain_intensity': 0.02,
            'color_temp': 1.0,
            'tint': 1.0,
            # New parameters
            'white_level': 240,
            'black_level': 16,
            'midtone_contrast': 1.0,
            'shadow_tint': (1.0, 1.0, 1.0),  # R,G,B tint for shadows (neutral)
            'highlight_tint': (1.0, 1.0, 1.0),  # R,G,B tint for highlights (neutral)
            'vignette_amount': 0.15,
            'sharpness': 0.15,
            'noise_reduction': 0.15,
        },
        'disgusted': {
            'contrast': 1.1,
            'saturation': 0.9,
            'shadow_lift': 1.1,
            'highlight_roll': 0.7,
            'color_intensity': 0.45,
            'halation_strength': 0.15,
            'grain_intensity': 0.02,
            'color_temp': 0.95,
            'tint': 1.05,
            # New parameters
            'white_level': 235,
            'black_level': 18,
            'midtone_contrast': 1.05,
            'shadow_tint': (0.95, 1.05, 0.95),  # R,G,B tint for shadows (greenish shadows)
            'highlight_tint': (0.95, 1.0, 0.95),  # R,G,B tint for highlights (slightly green highlights)
            'vignette_amount': 0.2,
            'sharpness': 0.2,
            'noise_reduction': 0.1,
        }
    }

def apply_primary_color_correction(frame, params):
    """Apply primary color correction to a frame"""
    # Convert to float32 for processing
    frame_float = frame.astype(np.float32) / 255.0
    
    # Adjust black and white levels
    frame_float = np.clip((frame_float - params['black_level']/255) * 
                          (255.0/(params['white_level'] - params['black_level'])), 0, 1)
    
    # Split channels for individual processing
    b, g, r = cv2.split(frame_float)
    
    # Apply contrast to each channel with different curves for shadows, midtones, and highlights
    def apply_contrast_curve(channel, contrast, mid_contrast):
        # Power function for general contrast
        channel = np.power(channel, contrast)
        
        # Apply midtone contrast (sigmoid-like function)
        midtones = 4 * (channel - 0.5) * (channel - 0.5)
        channel = channel + (mid_contrast - 1.0) * midtones * channel * (1 - channel)
        
        return np.clip(channel, 0, 1)
    
    r = apply_contrast_curve(r, params['contrast'], params['midtone_contrast'])
    g = apply_contrast_curve(g, params['contrast'], params['midtone_contrast'])
    b = apply_contrast_curve(b, params['contrast'], params['midtone_contrast'])
    
    # Color temperature adjustment
    temp_adjustment = np.array([params['color_temp'], 1.0, 1.0/params['color_temp']])
    tint_adjustment = np.array([1.0, params['tint'], 1.0])
    
    channels = np.stack([b, g, r], axis=2)
    
    # Apply temperature and tint
    channels = channels * np.reshape(temp_adjustment * tint_adjustment, (1, 1, 3))
    
    return np.clip(channels * 255, 0, 255).astype(np.uint8)


def apply_secondary_color_correction(frame, params, base_color):
    """Apply secondary color correction targeting specific color ranges"""
    # Convert to HSV for easier color targeting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Get the dominant hue from base_color to determine which colors to enhance
    if isinstance(base_color, str) and base_color.startswith('#'):
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        dominant_bgr = np.array([b, g, r])
    else:
        dominant_bgr = np.array(base_color)
    
    dominant_hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    dominant_hue = dominant_hsv[0]
    
    # Create a weight map for colors near the dominant hue
    hue_distance = np.minimum(np.abs(h - dominant_hue), 180 - np.abs(h - dominant_hue)) / 180.0
    weight_map = np.exp(-hue_distance * 4) * s / 255.0  # Weight by saturation too
    
    # Enhance saturation of similar colors
    s_enhancement = np.clip(s * (1 + weight_map * 0.3), 0, 255)
    
    # Adjust value of similar colors slightly
    v_enhancement = np.clip(v * (1 + weight_map * 0.1), 0, 255)
    
    # Put back together
    hsv_enhanced = cv2.merge([h, s_enhancement, v_enhancement])
    
    return cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_tint_by_luminance(frame, shadow_tint, highlight_tint):
    luminance = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    logger.debug(f"Mean luminance: {np.mean(luminance)}")
    
    shadow_mask = np.clip(1.0 - luminance * 2, 0, 1)
    highlight_mask = np.clip(luminance * 2 - 1.0, 0, 1)
    
    logger.debug(f"Mean shadow mask: {np.mean(shadow_mask)}")
    logger.debug(f"Mean highlight mask: {np.mean(highlight_mask)}")
    
    return frame  # Temporarily disable tinting to check if this is the issue


def apply_creative_grading(frame, params, base_color):
    """Apply creative look development and grading effects"""
    
    # Convert to float
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply tinting
    frame_float = apply_tint_by_luminance(frame_float, params['shadow_tint'], params['highlight_tint'])
    
    # Apply vignette
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.max(kernel)

    # Adjust vignette strength
    mask = np.power(mask / 255.0, 1 / (1 - params['vignette_amount'] * 0.3))  # Reduce strength
    mask = np.clip(mask, 0, 1)

    # Ensure vignette isn’t too dark
    if np.mean(mask) < 0.5:
        mask = np.clip(mask + 0.2, 0, 1)

    # Apply vignette effect more subtly
    frame_float = frame_float * (0.8 + 0.2 * np.expand_dims(mask, axis=2))

    # Convert back to uint8
    processed = np.clip(frame_float * 255, 0, 255).astype(np.uint8)

    return processed

def apply_sharpening(frame, strength):
    """Apply sharpening filter"""
    if strength <= 0:
        return frame
    
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]]) * strength + np.array([[0, 0, 0],
                                                               [0, 1, 0],
                                                               [0, 0, 0]]) * (1 - strength)
    
    return cv2.filter2D(frame, -1, kernel)

def reduce_noise(frame, strength):
    """Apply noise reduction"""
    if strength <= 0:
        return frame
    
    # Use a mix of bilateral filter (preserves edges) and gaussian blur
    h, w = frame.shape[:2]
    
    # Scale strength based on resolution (performance optimization)
    d = max(3, int(min(h, w) * 0.01 * strength))
    if d % 2 == 0:  # d must be odd
        d += 1
        
    # Apply bilateral filter for edge-preserving smoothing
    sigma_color = 10 * strength
    sigma_space = 10 * strength
    denoised = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    # For higher strength, blend with gaussian blur for smoother results
    if strength > 0.5:
        blur_amount = int(3 + 4 * (strength - 0.5))
        if blur_amount % 2 == 0:  # blur amount must be odd
            blur_amount += 1
        gaussian_blur = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
        blend_factor = (strength - 0.5) * 2  # 0 to 1 when strength is 0.5 to 1.0
        denoised = cv2.addWeighted(denoised, 1 - blend_factor, gaussian_blur, blend_factor, 0)
    
    return denoised

def create_halation(frame, strength, emotion_params):
    """Create emotion-aware halation effect"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = 200 + (emotion_params['highlight_roll'] - 0.75) * 50
    highlights = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    blur_amount = int(min(frame.shape[1], frame.shape[0]) * 0.03 * (1 + (emotion_params['halation_strength'] - 0.2)))
    if blur_amount % 2 == 0:
        blur_amount += 1
    halation = cv2.GaussianBlur(highlights, (blur_amount, blur_amount), 0)

    b_strength = 0.5 * emotion_params['color_temp']
    g_strength = 0.4 * emotion_params['tint']
    r_strength = 1.0 * (2 - emotion_params['color_temp'])

    halation_colored = cv2.merge([
        halation * b_strength,
        halation * g_strength,
        halation * r_strength
    ])

    return cv2.addWeighted(frame, 1, halation_colored.astype(np.uint8), strength * emotion_params['halation_strength'], 0)

def apply_grain(frame, emotion_params):
    """Applies emotion-specific film grain."""
    grain = np.random.normal(0, emotion_params['grain_intensity'], frame.shape).astype(np.float32)
    graded = np.clip(frame + grain * 255, 0, 255).astype(np.uint8)
    return graded

def analyze_frame_for_continuity(frame, reference_frame=None):
    """Analyze frame color information for continuity matching"""
    # Calculate histogram for comparison
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([frame_hsv], [0], None, [36], [0, 180])
    hist_s = cv2.calcHist([frame_hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([frame_hsv], [2], None, [32], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate average color
    avg_color = np.mean(frame, axis=(0, 1))
    
    # Calculate brightness and contrast metrics
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    frame_metrics = {
        'hist_h': hist_h,
        'hist_s': hist_s,
        'hist_v': hist_v,
        'avg_color': avg_color,
        'brightness': brightness,
        'contrast': contrast
    }
    
    # If we have a reference frame, calculate differences for matching
    if reference_frame is not None:
        ref_metrics = analyze_frame_for_continuity(reference_frame)
        
        h_diff = cv2.compareHist(frame_metrics['hist_h'], ref_metrics['hist_h'], cv2.HISTCMP_CORREL)
        s_diff = cv2.compareHist(frame_metrics['hist_s'], ref_metrics['hist_s'], cv2.HISTCMP_CORREL)
        v_diff = cv2.compareHist(frame_metrics['hist_v'], ref_metrics['hist_v'], cv2.HISTCMP_CORREL)
        
        color_diff = np.mean(np.abs(frame_metrics['avg_color'] - ref_metrics['avg_color']))
        brightness_diff = abs(frame_metrics['brightness'] - ref_metrics['brightness'])
        contrast_diff = abs(frame_metrics['contrast'] - ref_metrics['contrast'])
        
        # Calculate adjustments needed for matching
        match_adjustments = {
            'brightness_adj': ref_metrics['brightness'] - frame_metrics['brightness'],
            'contrast_adj': ref_metrics['contrast'] / max(0.1, frame_metrics['contrast']),
            'color_adj': ref_metrics['avg_color'] / np.maximum(frame_metrics['avg_color'], 0.1),
            'similarity': (h_diff + s_diff + v_diff) / 3  # Overall similarity score
        }
        
        frame_metrics['match_adjustments'] = match_adjustments
    
    return frame_metrics

def match_to_reference(frame, reference_metrics):
    """Adjust frame to match reference frame for continuity"""
    # Get adjustment parameters
    brightness_adj = reference_metrics['match_adjustments']['brightness_adj']
    contrast_adj = reference_metrics['match_adjustments']['contrast_adj']
    color_adj = reference_metrics['match_adjustments']['color_adj']
    
    # Apply adjustments
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply brightness adjustment
    frame_float = frame_float + (brightness_adj / 255.0)
    
    # Apply contrast adjustment (simplified)
    frame_float = (frame_float - 0.5) * contrast_adj + 0.5
    
    # Apply color adjustment
    b, g, r = cv2.split(frame_float)
    b = b * color_adj[0]
    g = g * color_adj[1]
    r = r * color_adj[2]
    frame_float = cv2.merge([b, g, r])
    
    # Convert back to uint8
    return np.clip(frame_float * 255, 0, 255).astype(np.uint8)

def create_emotion_lut(base_color, emotion_params):
    """Create emotion-specific LUT with film-like characteristics."""
    lut = np.zeros((256, 1), dtype=np.uint8)

    # Dynamically adjust parameters for balanced exposure
    shadow_lift = max(0.85, min(1.2, emotion_params['shadow_lift']))  
    highlight_roll = max(0.8, min(1.1, emotion_params['highlight_roll']))  
    contrast = max(0.9, min(1.15, emotion_params['contrast']))  
    saturation = max(0.9, min(1.1, emotion_params['saturation']))  
    color_intensity = emotion_params['color_intensity']

    logger.debug(f"LUT Params - Contrast: {contrast}, Saturation: {saturation}, Shadow Lift: {shadow_lift}, Highlight Roll: {highlight_roll}")

    for i in range(256):
        linear = (i / 255) ** 2.2  # Convert gamma to linear

        if linear < 0.04045:
            value = (linear / 12.92) * shadow_lift  # Lift shadows carefully
        else:
            value = (((linear + 0.055) / 1.055) ** 2.4) * contrast  # Adjust contrast while keeping details

        if value > highlight_roll:
            value = highlight_roll + (value - highlight_roll) * 0.4  # Smooth highlight transition, less aggressive

        # Ensure LUT does not make colors too dark or too bright
        value = min(0.95, max(0.2, value))  

        value *= saturation  # Apply moderate saturation adjustment
        display = value ** (1 / 2.2)  # Convert back to gamma space
        lut[i] = np.clip(display * 255, 0, 255)

    logger.debug(f"LUT Min: {np.min(lut)}, Max: {np.max(lut)}")
    return lut


def apply_emotion_grade(frame, emotion, color_bgr):
    """Applies emotion-specific grade."""
    emotion_params = enhance_color_grading_functions()[emotion]
    color_values = convert_hex_to_bgr(color_bgr)
    blue_lut = create_emotion_lut(color_values[0], emotion_params)
    green_lut = create_emotion_lut(color_values[1], emotion_params)
    red_lut = create_emotion_lut(color_values[2], emotion_params)

    b, g, r = cv2.split(frame.astype(np.float32) / 255)
    b = cv2.LUT((b * 255).astype(np.uint8), blue_lut).astype(np.float32) / 255
    g = cv2.LUT((g * 255).astype(np.uint8), green_lut).astype(np.float32) / 255
    r = cv2.LUT((r * 255).astype(np.uint8), red_lut).astype(np.float32) / 255
    
    graded = cv2.merge([b, g, r])
    graded = (graded * 255).astype(np.uint8)
    
    # Apply halation effect
    graded = create_halation(graded, 0.2, emotion_params)
    
    # Apply grain
    graded = apply_grain(graded, emotion_params)
    
    return graded

def detect_scenes(video_path, threshold=30):
    """Detect scene changes in a video by analyzing frame differences."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return [], []
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.debug(f"Analyzing video for scene changes: {total_frames} frames at {fps} fps")
        
        # Parameters for scene detection
        prev_frame = None
        scene_changes = []
        frame_count = 0
        
        # Process every Nth frame for efficiency
        sample_rate = max(1, int(fps / 2))  # Sample at half the framerate
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("End of video or failed to read frame.")
                break

            if frame is None or frame.size == 0:
                logger.error(f"Empty frame detected at {processed_frames}.")
                continue  # Skip processing this frame

            processed_frames += 1

            
            # Only process every Nth frame
            if frame_count % sample_rate != 0:
                continue

            
                
            # Resize for faster processing
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                frame_diff = cv2.absdiff(gray, prev_frame)
                non_zero_count = np.count_nonzero(frame_diff > threshold)
                
                # If more than 10% of pixels changed significantly, mark as scene change
                if non_zero_count > (320 * 180 * 0.1):
                    scene_changes.append(frame_count)
                    logger.debug(f"Scene change detected at frame {frame_count}")
            
            prev_frame = gray
            
        cap.release()
        
        # Convert scene changes to time values
        scene_times = [t / fps for t in scene_changes]
        
        # Filter out scenes that are too close together (less than 1 second)
        filtered_scenes = [scene_changes[0]] if scene_changes else []
        filtered_times = [scene_times[0]] if scene_times else []
        
        for i in range(1, len(scene_changes)):
            if scene_times[i] - filtered_times[-1] >= 1.0:  # At least 1 second apart
                filtered_scenes.append(scene_changes[i])
                filtered_times.append(scene_times[i])
        
        # Always include the start
        if len(filtered_scenes) == 0 or filtered_scenes[0] > 0:
            filtered_scenes.insert(0, 0)
            filtered_times.insert(0, 0.0)
            
        logger.debug(f"Detected {len(filtered_scenes)} scenes")
        return filtered_scenes, filtered_times
        
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        return [0], [0.0]  # Return default if failure

def apply_cinema_grade_color(input_path, output_path, color_hex):
    """Apply cinema-grade color correction to video based on emotion."""
    try:
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open input video: {input_path}")
            return {"error": "Failed to open input video"}

        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from video")
            return {"error": "Failed to read frame"}

        # Detect emotion from the first frame
        emotion = detect_emotion(frame)  
        logger.debug(f"Detected emotion: {emotion} for color: {color_hex}")
      
        
        # Detect scene changes for better color grading continuity
        scene_frames, scene_times = detect_scenes(input_path)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open input video: {input_path}")
            return {"error": "Failed to open input video"}
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.debug(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        logger.debug(f"Output VideoWriter initialized: {width}x{height}, FPS={fps}")

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            return {"error": "Failed to create output video"}
            
        # Initialize variables for progress tracking
        processed_frames = 0
        current_scene = 0
        scene_reference_frames = {}
        color_bgr = convert_hex_to_bgr(color_hex)
        
        # Emotion parameters for grading
        emotion_params = enhance_color_grading_functions()[emotion]
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Check if we're entering a new scene
            if current_scene + 1 < len(scene_frames) and processed_frames >= scene_frames[current_scene + 1]:
                current_scene += 1
                logger.debug(f"Processing scene {current_scene} starting at frame {processed_frames}")
            
            # Process the frame with our color grading pipeline
            processed = frame.copy()
            
            # Step 1: Apply primary color correction
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "step_1_original.jpg"), frame)

            processed = apply_primary_color_correction(frame, emotion_params)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "step_2_primary_correction.jpg"), processed)

            processed = apply_secondary_color_correction(processed, emotion_params, color_bgr)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "step_3_secondary_correction.jpg"), processed)

            processed = apply_creative_grading(processed, emotion_params, color_bgr)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "step_4_creative_grading.jpg"), processed)

            

            # Step 4: Apply sharpening
            processed = apply_sharpening(processed, emotion_params.get('sharpness', 0.15))
            
            # Step 5: Apply noise reduction
            processed = reduce_noise(processed, emotion_params.get('noise_reduction', 0.15))
            
            # Step 6: Apply emotion-specific look
            processed = apply_emotion_grade(processed, emotion, color_bgr)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "step_5_emotion_grading.jpg"), processed)

            if processed_frames == 2:  # Save only the first frame for testing
                cv2.imwrite(os.path.join(UPLOAD_FOLDER, "test_frame.jpg"), frame)

            
            # Scene continuity management
            if processed_frames == scene_frames[current_scene]:
                # Save reference frame for this scene
                scene_reference_frames[current_scene] = analyze_frame_for_continuity(processed)
            elif current_scene > 0 and processed_frames == scene_frames[current_scene] + 1:
                # For the second frame in a new scene, match to previous scene end for continuity
                prev_scene = current_scene - 1
                if prev_scene in scene_reference_frames:
                    # Analyze current frame
                    current_metrics = analyze_frame_for_continuity(processed, None)
                    # Add match adjustment data compared to previous scene
                    prev_reference = scene_reference_frames[prev_scene]
                    
                    # Calculate adjustments for continuity (50% strength for natural transition)
                    h_diff = cv2.compareHist(current_metrics['hist_h'], prev_reference['hist_h'], cv2.HISTCMP_CORREL)
                    s_diff = cv2.compareHist(current_metrics['hist_s'], prev_reference['hist_s'], cv2.HISTCMP_CORREL)
                    v_diff = cv2.compareHist(current_metrics['hist_v'], prev_reference['hist_v'], cv2.HISTCMP_CORREL)
                    
                    # If scenes are very different, don't force continuity
                    similarity = (h_diff + s_diff + v_diff) / 3
                    if similarity < 0.5:  # Scenes differ a lot
                        logger.debug(f"Scenes too different (similarity: {similarity:.2f}), not forcing continuity")
                    else:
                        # Apply soft matching for subtle continuity
                        brightness_adj = (prev_reference['brightness'] - current_metrics['brightness']) * 0.3
                        contrast_ratio = max(0.8, min(1.2, prev_reference['contrast'] / max(0.1, current_metrics['contrast'])))
                        
                        # Apply subtle adjustments
                        processed_float = processed.astype(np.float32) / 255.0
                        processed_float = processed_float + (brightness_adj / 255.0)
                        processed_float = (processed_float - 0.5) * contrast_ratio + 0.5
                        processed = np.clip(processed_float * 255, 0, 255).astype(np.uint8)
            if processed is None or processed.size == 0:
                    logger.error(f"Processed frame at {processed_frames} is empty!")
                    continue
            # Save a sample frame to check visually
            if processed_frames == 10:  # Save the 10th processed frame for testing
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "processed_test_frame.jpg"), processed)

            if processed.dtype != np.uint8:
                    logger.warning(f"Processed frame at {processed_frames} is not uint8, converting...")
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Write processed frame to output
            out.write(processed)
            
            # Log progress
            if processed_frames % 100 == 0 or processed_frames == frame_count:
                progress = processed_frames / frame_count * 100
                logger.debug(f"Processing progress: {progress:.1f}% ({processed_frames}/{frame_count})")
        
        # Release resources
        cap.release()
        out.release()
        
        logger.debug(f"Video processing completed: {output_path}")
        
        # Return information about the processed video
        return {
            "output_path": output_path,
            "detected_emotion": emotion,
            "color_hex": color_hex,
            "detected_scenes": len(scene_frames),
            "scene_start_frames": scene_frames,
            "scene_start_times": scene_times,
            "total_frames": processed_frames
        }
        
    except Exception as e:
        logger.error(f"Error in cinema grade color processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == '__main__':
    # Set up proper logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app.run(debug=True, host='0.0.0.0', port=5000)