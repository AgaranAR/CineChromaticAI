"""
app.py — Cinematic emotion grading pipeline (act-aware, temporally evolving)

Pipeline:
  1. script.py → 3 acts, each with emotion + tension (0.0–1.0)
  2. Each act maps to a grade profile
  3. Tension modulates vignette + grain ON TOP of the base profile
  4. Grades interpolate smoothly between act boundaries
     so the visual tone evolves with the narrative arc
"""

import os
import time
import logging
import threading
import json

import cv2
import numpy as np
import pdfplumber
from flask import Flask, jsonify, render_template, request, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename

import script

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-in-prod")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"txt", "pdf", "mp4", "mov", "avi"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_progress: dict = {}
_progress_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Base grade profiles — 4 parameters, each with a clear rationale
# ---------------------------------------------------------------------------
GRADE_PROFILES = {
    "happy": {
        "brightness_offset":  0.06,
        "saturation_scale":   1.20,
        "color_temp":         1.08,   # warm gold
        "vignette_strength":  0.10,
    },
    "sad": {
        "brightness_offset": -0.08,
        "saturation_scale":   0.75,
        "color_temp":         0.90,   # cool blue
        "vignette_strength":  0.35,
    },
    "angry": {
        "brightness_offset":  0.04,
        "saturation_scale":   0.85,
        "color_temp":         1.12,   # hot red-orange
        "vignette_strength":  0.35,
    },
    "fearful": {
        "brightness_offset": -0.05,
        "saturation_scale":   0.82,
        "color_temp":         0.87,   # cold blue-grey
        "vignette_strength":  0.40,
    },
    "neutral": {
        "brightness_offset":  0.0,
        "saturation_scale":   1.0,
        "color_temp":         1.0,
        "vignette_strength":  0.08,
    },
}


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _lerp_profiles(a: dict, b: dict, t: float) -> dict:
    """
    Linearly interpolate between two grade profiles.
    t=0.0 → profile a, t=1.0 → profile b.
    """
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def _apply_tension(profile: dict, tension: float) -> dict:
    """
    Tension (0.0–1.0) pushes the grade further into its extreme:
      - vignette gets heavier  (up to +0.25 at full tension)
      - brightness drops more  (up to -0.04 at full tension)
      - saturation drops more  (up to -0.10 at full tension)
      - grain intensity scales with tension (returned separately)

    Does NOT change color_temp — the emotional temperature is set by the
    base profile, tension just deepens the darkness and isolation.
    """
    p = dict(profile)
    p["vignette_strength"]  = min(p["vignette_strength"]  + tension * 0.25, 0.75)
    p["brightness_offset"]  = max(p["brightness_offset"]  - tension * 0.04, -0.20)
    p["saturation_scale"]   = max(p["saturation_scale"]   - tension * 0.10,  0.60)
    return p


def _build_frame_profiles(acts: list, total_frames: int) -> list:
    """
    Pre-compute the grade profile for every frame in the video.

    Within each act the profile is held constant.
    At act boundaries a smooth crossfade spans TRANSITION_FRAMES frames
    so there's no hard jump between acts.

    Returns a list of length total_frames, where each entry is a dict
    with the interpolated + tension-modulated grade parameters for that frame,
    plus "grain_intensity" derived from tension.
    """
    TRANSITION_FRAMES = max(30, total_frames // 20)  # ~1s at 30fps, max 5% of video

    # Map each act to its final (tension-adjusted) profile
    act_profiles = []
    for act in acts:
        base    = GRADE_PROFILES.get(act["emotion"], GRADE_PROFILES["neutral"])
        tensioned = _apply_tension(base, act["tension"])
        tensioned["grain_intensity"] = act["tension"] * 0.025  # 0 at calm, 0.025 at peak
        act_profiles.append(tensioned)

    # Compute act frame boundaries
    act_boundaries = []
    for act in acts:
        start_frame = int(act["start"] * total_frames)
        end_frame   = int(act["end"]   * total_frames)
        act_boundaries.append((start_frame, end_frame))

    # Build per-frame profile list
    profiles = []
    for frame_idx in range(total_frames):
        # Find which act this frame is in
        current_act_idx = 0
        for i, (start, end) in enumerate(act_boundaries):
            if frame_idx >= start:
                current_act_idx = i

        next_act_idx = min(current_act_idx + 1, len(act_profiles) - 1)
        _, current_end = act_boundaries[current_act_idx]

        # Are we inside a transition zone? (approaching the next act boundary)
        frames_until_boundary = current_end - frame_idx
        if frames_until_boundary <= TRANSITION_FRAMES and current_act_idx < len(act_profiles) - 1:
            # Smooth lerp into the next act
            t = 1.0 - (frames_until_boundary / TRANSITION_FRAMES)
            t = t * t * (3 - 2 * t)  # smoothstep — ease in/out, not linear
            profile = _lerp_profiles(act_profiles[current_act_idx],
                                     act_profiles[next_act_idx], t)
        else:
            profile = act_profiles[current_act_idx]

        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# Vignette mask cache
# ---------------------------------------------------------------------------
_vignette_cache: dict = {}

def _get_vignette_mask(height: int, width: int, strength: float) -> np.ndarray:
    key = (height, width, round(strength, 3))
    if key not in _vignette_cache:
        cx, cy = width / 2, height / 2
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        mask = 1.0 - strength * np.clip(dist, 0, 1) ** 1.5
        _vignette_cache[key] = mask.astype(np.float32)
    return _vignette_cache[key]


# ---------------------------------------------------------------------------
# Frame transforms — each has one job
# ---------------------------------------------------------------------------

def apply_levels_and_tint(frame: np.ndarray, params: dict) -> np.ndarray:
    """Saturation, brightness, color temperature — midtone-masked."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["saturation_scale"], 0, 255)
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    f = frame.astype(np.float32) / 255.0
    temp = params["color_temp"]

    # Luminance mask: protect highlights and shadows, apply temp mostly to midtones
    lum  = 0.299 * f[:,:,2] + 0.587 * f[:,:,1] + 0.114 * f[:,:,0]
    mask = np.clip(1.0 - np.abs(lum - 0.5) * 2.5, 0, 1)

    f[:,:,2] = np.clip(f[:,:,2] * (1.0 + (temp - 1.0) * mask), 0, 1)          # R
    f[:,:,0] = np.clip(f[:,:,0] * (1.0 + (1.0/max(temp, 0.01) - 1.0) * mask), 0, 1)  # B
    f = np.clip(f + params["brightness_offset"], 0, 1)

    return (f * 255).astype(np.uint8)


def apply_vignette(frame: np.ndarray, strength: float) -> np.ndarray:
    """Radial edge darkening. strength=0 → no effect."""
    if strength <= 0.0:
        return frame
    h, w = frame.shape[:2]
    mask = _get_vignette_mask(h, w, strength)
    f = frame.astype(np.float32) / 255.0
    f *= mask[:, :, np.newaxis]
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def apply_grain(frame: np.ndarray, intensity: float) -> np.ndarray:
    """Film grain — zero-mean Gaussian noise."""
    if intensity <= 0.0:
        return frame
    noise = np.random.normal(0, intensity, frame.shape).astype(np.float32)
    f = frame.astype(np.float32) / 255.0
    return np.clip((f + noise) * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Core video processor
# ---------------------------------------------------------------------------

def process_video(input_path: str, output_path: str, acts: list,
                  job_id: str = None) -> dict:
    """
    Grade each frame using its act's profile, interpolated at boundaries.

    Args:
        input_path  : source video
        output_path : output path
        acts        : list of act dicts from script.extract_acts()
        job_id      : optional — if set, real progress pushed to _progress

    Returns dict with output_path, total_frames, and act summary.
    """
    def _set_progress(pct: int, done: bool = False, error: str = None):
        if job_id:
            with _progress_lock:
                _progress[job_id] = {"pct": pct, "done": done, "error": error}

    _set_progress(0)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        msg = f"Cannot open video: {input_path}"
        _set_progress(0, done=True, error=msg)
        return {"error": msg}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    # Pre-compute all per-frame profiles (cheap — just dicts)
    frame_profiles = _build_frame_profiles(acts, total)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        msg = f"Cannot create output: {output_path}"
        _set_progress(0, done=True, error=msg)
        return {"error": msg}

    processed = 0
    last_pct  = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        params = frame_profiles[min(processed, total - 1)]

        frame = apply_levels_and_tint(frame, params)
        frame = apply_vignette(frame, params["vignette_strength"])
        if params.get("grain_intensity", 0) > 0.002:
            frame = apply_grain(frame, params["grain_intensity"])

        out.write(frame)
        processed += 1

        if total > 0:
            pct = int(processed / total * 100)
            if pct != last_pct:
                _set_progress(pct)
                last_pct = pct

    cap.release()
    out.release()

    _set_progress(100, done=True)
    logger.debug(f"Done: {processed} frames → {output_path}")

    return {
        "output_path":  output_path,
        "total_frames": processed,
        "acts": [
            {"act": a["act"], "emotion": a["emotion"], "tension": a["tension"]}
            for a in acts
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        if "script" not in request.files or "video" not in request.files:
            return jsonify({"error": "Both a script file and a video file are required"}), 400

        script_file = request.files["script"]
        video_file  = request.files["video"]

        if not script_file.filename or not video_file.filename:
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(script_file.filename) or not allowed_file(video_file.filename):
            return jsonify({"error": "Invalid file type. Allowed: txt, pdf, mp4, mov, avi"}), 400

        script_path = os.path.join(UPLOAD_FOLDER, secure_filename(script_file.filename))
        video_path  = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))
        script_file.save(script_path)
        video_file.save(video_path)

        # Extract script text
        if script_path.endswith(".pdf"):
            script_text = extract_text_from_pdf(script_path)
        else:
            with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
                script_text = f.read()

        if not script_text.strip():
            return jsonify({"error": "Could not extract text from the script file"}), 400

        # Stage 1: resolve acts from script
        acts = script.extract_acts(script_text)
        logger.debug(f"Acts resolved: {acts}")

        # Stage 2+3: grade video in background thread
        job_id = f"job_{int(time.time() * 1000)}"
        output_filename = f"graded_{os.path.splitext(secure_filename(video_file.filename))[0]}.mp4"
        output_path     = os.path.join(UPLOAD_FOLDER, output_filename)

        with _progress_lock:
            _progress[job_id] = {"pct": 0, "done": False, "error": None}

        def _run():
            result = process_video(video_path, output_path, acts, job_id=job_id)
            for path in (script_path, video_path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Could not delete {path}: {e}")
            if "error" in result:
                with _progress_lock:
                    _progress[job_id]["error"] = result["error"]

        threading.Thread(target=_run, daemon=True).start()

        return jsonify({
            "job_id":       job_id,
            "output_video": output_filename,
            "acts": [
                {
                    "act":     a["act"],
                    "emotion": a["emotion"],
                    "tension": round(a["tension"], 2),
                    "video_segment": f"{a['start']:.0%}–{a['end']:.0%}",
                }
                for a in acts
            ],
        })

    except Exception as e:
        logger.exception("Unhandled error in /process")
        return jsonify({"error": str(e)}), 500


@app.route("/progress/<job_id>")
def progress(job_id):
    """Server-Sent Events — real frame-by-frame progress."""
    def _stream():
        while True:
            with _progress_lock:
                state = _progress.get(job_id)
            if state is None:
                yield f"data: {json.dumps({'error': 'unknown job'})}\n\n"
                break
            yield f"data: {json.dumps(state)}\n\n"
            if state["done"] or state["error"]:
                def _cleanup():
                    time.sleep(30)
                    with _progress_lock:
                        _progress.pop(job_id, None)
                threading.Thread(target=_cleanup, daemon=True).start()
                break
            time.sleep(0.5)

    return Response(
        stream_with_context(_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<path:filename>")
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    for _ in range(5):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return send_file(file_path, as_attachment=True)
        time.sleep(1)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
