"""Golf swing analysis utilities using pose-estimation landmarks.

This module provides two layers:
- Landmark-driven analysis that accepts normalized pose coordinates for unit testing.
- A video-first helper that extracts pose landmarks with MediaPipe before analysis.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


Point = Tuple[float, float]
LandmarkFrame = Mapping[str, Point]


@dataclass
class _AngleStats:
    values: List[float]

    def model(self) -> "AngleWindow":
        if not self.values:
            return AngleWindow(min_deg=0.0, max_deg=0.0, range_deg=0.0)

        min_v = min(self.values)
        max_v = max(self.values)
        return AngleWindow(min_deg=min_v, max_deg=max_v, range_deg=max_v - min_v)


class AngleWindow(BaseModel):
    """Simple container for min/max/range degrees."""

    min_deg: float
    max_deg: float
    range_deg: float


class SwingPhaseTiming(BaseModel):
    """Backswing/downswing timing measurements."""

    backswing_ms: float
    downswing_ms: float
    tempo_ratio: float


class SwingAnalysisResult(BaseModel):
    total_frames: int
    fps: float
    frames_with_pose: int
    shoulder_rotation: AngleWindow
    hip_rotation: AngleWindow
    torso_tilt: AngleWindow
    lead_wrist_height: AngleWindow
    peak_hand_speed: float = Field(
        ..., description="Peak lead-hand speed normalized by torso length (units/sec)"
    )
    phase_timing: SwingPhaseTiming
    notes: List[str]


_LANDMARK_KEYS = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_wrist",
    "right_wrist",
}


def _midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _angle_deg(p1: Point, p2: Point) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def _vertical_angle_deg(origin: Point, target: Point) -> float:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return math.degrees(math.atan2(dx, -dy))


def _torso_length(shoulder_mid: Point, hip_mid: Point) -> float:
    length = math.dist(shoulder_mid, hip_mid)
    return max(length, 1e-6)


def _select_lead_wrist(frame: LandmarkFrame, lead_hand: str) -> Point:
    if lead_hand.lower() == "right":
        return frame["right_wrist"]
    return frame["left_wrist"]


def analyze_landmark_sequence(
    landmarks: Sequence[LandmarkFrame],
    fps: float,
    *,
    lead_hand: str = "left",
) -> SwingAnalysisResult:
    """Compute swing metrics from normalized landmark frames.

    Args:
        landmarks: Iterable of frames, each providing normalized (x, y) tuples
            for the required keys.
        fps: Frames-per-second for timing calculations.
        lead_hand: "left" or "right" to identify the lead wrist.
    """

    total_frames = len(landmarks)
    if total_frames == 0:
        raise ValueError("No landmark frames provided.")

    fps = fps if fps > 1e-3 else 30.0

    shoulder_angles: List[float] = []
    hip_angles: List[float] = []
    tilt_angles: List[float] = []
    wrist_heights: List[float] = []
    hand_speeds: List[float] = []
    wrist_x_positions: List[float] = []

    frames_with_pose = 0
    torso_len_ref = None

    prev_wrist: Point | None = None

    for frame in landmarks:
        if not _LANDMARK_KEYS.issubset(frame.keys()):
            continue

        ls = frame["left_shoulder"]
        rs = frame["right_shoulder"]
        lh = frame["left_hip"]
        rh = frame["right_hip"]

        lead_wrist = _select_lead_wrist(frame, lead_hand)

        shoulder_mid = _midpoint(ls, rs)
        hip_mid = _midpoint(lh, rh)

        torso_len = _torso_length(shoulder_mid, hip_mid)
        torso_len_ref = torso_len_ref or torso_len

        shoulder_angles.append(_angle_deg(ls, rs))
        hip_angles.append(_angle_deg(lh, rh))
        tilt_angles.append(_vertical_angle_deg(hip_mid, shoulder_mid))
        wrist_heights.append((shoulder_mid[1] - lead_wrist[1]) / torso_len)
        wrist_x_positions.append(lead_wrist[0])

        if prev_wrist is not None:
            dist = math.dist(prev_wrist, lead_wrist)
            hand_speeds.append((dist / torso_len) * fps)
        prev_wrist = lead_wrist

        frames_with_pose += 1

    if frames_with_pose == 0:
        raise ValueError("No frames contained full pose landmarks.")

    peak_speed = max(hand_speeds) if hand_speeds else 0.0

    phase_timing = _phase_timings(wrist_x_positions, fps)

    notes = _build_notes(peak_speed, phase_timing.tempo_ratio, tilt_angles)

    return SwingAnalysisResult(
        total_frames=total_frames,
        fps=fps,
        frames_with_pose=frames_with_pose,
        shoulder_rotation=_AngleStats(shoulder_angles).model(),
        hip_rotation=_AngleStats(hip_angles).model(),
        torso_tilt=_AngleStats(tilt_angles).model(),
        lead_wrist_height=_AngleStats(wrist_heights).model(),
        peak_hand_speed=peak_speed,
        phase_timing=phase_timing,
        notes=notes,
    )


def _phase_timings(wrist_x_positions: Sequence[float], fps: float) -> SwingPhaseTiming:
    if len(wrist_x_positions) < 2:
        return SwingPhaseTiming(backswing_ms=0.0, downswing_ms=0.0, tempo_ratio=0.0)

    start_x = wrist_x_positions[0]
    extremum_idx = int(np.argmax(np.abs(np.array(wrist_x_positions) - start_x)))

    backswing_frames = max(extremum_idx, 1)
    downswing_frames = max(len(wrist_x_positions) - extremum_idx, 1)

    backswing_ms = backswing_frames / fps * 1000.0
    downswing_ms = downswing_frames / fps * 1000.0

    tempo_ratio = backswing_ms / downswing_ms if downswing_ms > 1e-6 else 0.0

    return SwingPhaseTiming(
        backswing_ms=backswing_ms,
        downswing_ms=downswing_ms,
        tempo_ratio=tempo_ratio,
    )


def _build_notes(peak_speed: float, tempo_ratio: float, tilt_angles: Iterable[float]) -> List[str]:
    notes = []

    if peak_speed < 1.2:
        notes.append("Tempo is smooth; consider adding speed through impact.")
    elif peak_speed > 4.0:
        notes.append("Hand speed peaks high—ensure balance and sequencing.")

    if 1.8 <= tempo_ratio <= 2.2:
        notes.append("Tempo ratio near the tour average of ~2.0:1.")
    else:
        notes.append("Work toward a ~2.0:1 backswing/downswing tempo.")

    if tilt_angles:
        tilt_range = max(tilt_angles) - min(tilt_angles)
        if tilt_range < 5:
            notes.append("Upper body stayed level—maintain athletic spine tilt.")
        elif tilt_range > 20:
            notes.append("Large spine-tilt changes detected; monitor posture sway.")

    return notes


def analyze_swing_video(
    video_bytes: bytes,
    *,
    lead_hand: str = "left",
) -> SwingAnalysisResult:
    """Extract pose landmarks from video bytes and compute swing metrics."""

    if not video_bytes:
        raise ValueError("Video payload is empty.")

    import mediapipe as mp

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    try:
        if not cap.isOpened():
            raise ValueError("Unable to read video stream.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if fps > 1e-3 else 30.0

        pose = mp.solutions.pose.Pose(model_complexity=1, enable_segmentation=False)
        frames: List[LandmarkFrame] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                frames.append({})
                continue

            coords: Dict[str, Point] = {}
            landmarks = result.pose_landmarks.landmark

            coords["left_shoulder"] = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                                        landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y)
            coords["right_shoulder"] = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                                         landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y)
            coords["left_hip"] = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
                                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
            coords["right_hip"] = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
                                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y)
            coords["left_wrist"] = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y)
            coords["right_wrist"] = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y)

            frames.append(coords)

        if not frames:
            raise ValueError("Uploaded file contained no readable frames.")

        return analyze_landmark_sequence(frames, fps, lead_hand=lead_hand)
    finally:
        cap.release()
        os.remove(temp_path)
