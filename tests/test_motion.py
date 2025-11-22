"""Unit tests for computer-vision utilities."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from app.cv_utils import MotionDetectionResult, detect_motion_segments


def _build_synthetic_motion_video(frame_count: int = 15) -> bytes:
    """Create a small AVI video with a moving square."""

    height, width = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fd, path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)

    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))

    try:
        for idx in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if 5 <= idx < 10:
                cv2.rectangle(frame, (10 + idx, 10), (30 + idx, 30), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()

    with open(path, "rb") as fh:
        data = fh.read()

    os.remove(path)

    return data


class TestMotionDetection(unittest.TestCase):
    def test_detects_motion_segment(self) -> None:
        payload = _build_synthetic_motion_video()

        result: MotionDetectionResult = detect_motion_segments(
            payload,
            min_area=20,
            intensity_threshold=8,
            movement_ratio_threshold=0.01,
            calm_down_frames=1,
        )

        self.assertEqual(result.total_frames, 15)
        self.assertGreater(result.movement_frames, 0)
        self.assertTrue(result.segments)

        first_segment = result.segments[0]
        self.assertLessEqual(first_segment.start_frame, 5)
        self.assertGreaterEqual(first_segment.end_frame, 9)

        self.assertGreater(result.movement_ratio, 0.05)
        self.assertLess(result.movement_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()

