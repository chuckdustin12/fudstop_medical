"""Computer-vision utilities for motion awareness.

This module focuses on lightweight human-movement detection. It performs
frame differencing to highlight motion intensity, then groups consecutive
movement frames into labeled segments. The logic is intentionally
stateless and parameterized so it can be reused by FastAPI routes and
unit tests alike.
"""

from typing import List, Sequence

from pydantic import BaseModel


class MotionSegment(BaseModel):
    """A contiguous period of motion in the video."""

    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    avg_movement: float


class MotionDetectionResult(BaseModel):
    """Summary statistics for detected movement."""

    total_frames: int
    fps: float
    movement_frames: int
    movement_ratio: float
    segments: Sequence[MotionSegment]


def _finalize_segment(
    segments: List[MotionSegment],
    fps: float,
    start_frame: int,
    end_frame: int,
    movement_scores: List[float],
) -> None:
    """Append a completed motion segment with timing metadata."""

    if not movement_scores:
        movement_scores = [0.0]

    avg_movement = sum(movement_scores) / len(movement_scores)
    start_ms = (start_frame / fps) * 1000.0
    end_ms = (end_frame / fps) * 1000.0

    segments.append(
        MotionSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            avg_movement=avg_movement,
        )
    )


def detect_motion_segments(
    video_bytes: bytes,
    *,
    min_area: int = 500,
    intensity_threshold: int = 18,
    movement_ratio_threshold: float = 0.02,
    calm_down_frames: int = 3,
) -> MotionDetectionResult:
    """Detect human body movement periods in a short video clip.

    The algorithm:
    1. Writes the upload to a temp file for OpenCV ingestion.
    2. Uses frame differencing to build binary masks of movement.
    3. Scores each frame by percentage of moving pixels and contour area.
    4. Groups consecutive moving frames into labeled segments.
    """

    import os
    import tempfile

    import cv2
    import numpy as np

    if not video_bytes:
        raise ValueError("Video payload is empty.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    try:
        if not cap.isOpened():
            raise ValueError("Unable to read video stream.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if fps > 1e-3 else 30.0

        total_frames = 0
        movement_frames = 0
        segments: List[MotionSegment] = []

        prev_gray = None
        segment_start = None
        segment_scores: List[float] = []
        last_motion_frame = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            frame_idx = total_frames - 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            delta = cv2.absdiff(prev_gray, gray)
            prev_gray = gray

            _, thresh = cv2.threshold(delta, intensity_threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = max((cv2.contourArea(c) for c in contours), default=0.0)

            movement_score = float(np.sum(dilated) / (255.0 * dilated.size))
            moving = movement_score >= movement_ratio_threshold or max_area >= min_area

            if moving:
                movement_frames += 1
                last_motion_frame = frame_idx
                if segment_start is None:
                    segment_start = frame_idx
                segment_scores.append(movement_score)
            elif segment_start is not None and frame_idx - last_motion_frame > calm_down_frames:
                _finalize_segment(segments, fps, segment_start, last_motion_frame, segment_scores)
                segment_start = None
                segment_scores = []

        if total_frames == 0:
            raise ValueError("Uploaded file contained no readable frames.")

        if segment_start is not None:
            _finalize_segment(segments, fps, segment_start, last_motion_frame, segment_scores)

        movement_ratio = movement_frames / max(total_frames, 1)

        return MotionDetectionResult(
            total_frames=total_frames,
            fps=fps,
            movement_frames=movement_frames,
            movement_ratio=movement_ratio,
            segments=segments,
        )
    finally:
        cap.release()
        os.remove(temp_path)
