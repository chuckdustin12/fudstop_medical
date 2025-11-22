"""Unit tests for swing analysis utilities."""

import unittest

from app.swing_utils import SwingAnalysisResult, analyze_landmark_sequence


def _synthetic_frame(x_offset: float, wrist_height: float = 0.0) -> dict[str, tuple[float, float]]:
    """Build a simple pose frame with adjustable lead-wrist position."""

    ls = (0.4, 0.4)
    rs = (0.6, 0.4)
    lh = (0.42, 0.6)
    rh = (0.58, 0.6)

    lead_wrist = (0.5 + x_offset, 0.55 - wrist_height)

    return {
        "left_shoulder": ls,
        "right_shoulder": rs,
        "left_hip": lh,
        "right_hip": rh,
        "left_wrist": lead_wrist,
        "right_wrist": (1 - lead_wrist[0], lead_wrist[1]),
    }


class TestSwingAnalysis(unittest.TestCase):
    def test_landmark_sequence_metrics(self) -> None:
        frames = []
        for idx in range(20):
            # Backswing: move away from target (negative x), then return through the ball.
            if idx < 10:
                x_offset = -0.02 * idx
            else:
                x_offset = -0.2 + 0.025 * (idx - 10)

            wrist_height = 0.02 if 5 <= idx <= 12 else 0.0
            frames.append(_synthetic_frame(x_offset, wrist_height))

        result: SwingAnalysisResult = analyze_landmark_sequence(frames, fps=30.0, lead_hand="left")

        self.assertEqual(result.total_frames, 20)
        self.assertEqual(result.frames_with_pose, 20)
        self.assertGreater(result.phase_timing.backswing_ms, 0)
        self.assertGreater(result.phase_timing.downswing_ms, 0)
        self.assertGreater(result.peak_hand_speed, 0)
        self.assertGreater(result.lead_wrist_height.range_deg, 0)

        self.assertTrue(result.notes)
        self.assertAlmostEqual(result.phase_timing.tempo_ratio, 1.0, delta=1.2)


if __name__ == "__main__":
    unittest.main()
