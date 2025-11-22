"""Routes for computer-vision powered assessments."""

import logging
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.cv_utils import MotionDetectionResult, detect_motion_segments
from app.swing_utils import SwingAnalysisResult, analyze_swing_video

visionrouter = APIRouter()
logger = logging.getLogger("vision")


@visionrouter.post("/motion/detect", response_model=MotionDetectionResult)
async def detect_motion(
    file: UploadFile = File(...),
    min_area: int = 250,
    intensity_threshold: int = 14,
    movement_ratio_threshold: float = 0.02,
    calm_down_frames: int = 3,
) -> MotionDetectionResult:
    """Detect human movement segments in an uploaded video clip."""

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Upload a .mp4, .mov, or .avi video.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    try:
        result = detect_motion_segments(
            payload,
            min_area=min_area,
            intensity_threshold=intensity_threshold,
            movement_ratio_threshold=movement_ratio_threshold,
            calm_down_frames=calm_down_frames,
        )
    except ValueError as exc:
        logger.warning("Motion detection failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    return result


@visionrouter.get("/motion/config")
async def motion_config() -> Dict[str, str]:
    """Expose key tunables for client-side discoverability."""

    return {
        "description": "Lightweight frame-difference motion detector tuned for human movement.",
        "fields": {
            "min_area": "Minimum contour area (px^2) for movement to be counted",
            "intensity_threshold": "Pixel delta (0-255) treated as movement",
            "movement_ratio_threshold": "Percentage of moving pixels to flag a frame",
            "calm_down_frames": "Frames of stillness before closing a segment",
        },
    }


@visionrouter.post("/swing/analyze", response_model=SwingAnalysisResult)
async def analyze_swing(
    file: UploadFile = File(...),
    lead_hand: str = "left",
) -> SwingAnalysisResult:
    """Analyze a golf swing and return pose-derived swing metrics."""

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Upload a .mp4, .mov, or .avi video.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    try:
        return analyze_swing_video(payload, lead_hand=lead_hand)
    except ValueError as exc:
        logger.warning("Swing analysis failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@visionrouter.get("/swing/config")
async def swing_config() -> Dict[str, Dict[str, str]]:
    """Document swing-analysis inputs and output fields for UI clients."""

    return {
        "description": "Pose-estimation driven golf-swing analyzer returning tempo and posture metrics.",
        "inputs": {
            "file": "Upload an .mp4/.mov/.avi side-on swing clip.",
            "lead_hand": """Use 'left' (default) or 'right' to identify the lead wrist for tempo/speed calculations.""",
        },
        "outputs": {
            "shoulder_rotation": "Min/max/range of shoulder line angle in degrees.",
            "hip_rotation": "Min/max/range of hip line angle in degrees.",
            "torso_tilt": "Min/max/range of spine tilt relative to vertical (degrees).",
            "lead_wrist_height": "Normalized lead hand height relative to shoulders/hips.",
            "peak_hand_speed": "Peak lead-hand speed normalized by torso length (units/sec).",
            "phase_timing": "Backswing/downswing durations (ms) and tempo ratio.",
            "notes": "Coach-style guidance extracted from the measured metrics.",
        },
    }

