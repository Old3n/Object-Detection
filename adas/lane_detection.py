"""Classical image-space lane-boundary estimation for ADAS research.

This is a lightweight baseline based on lane-marking color/edge cues and Hough
segments. It is not a learned lane-segmentation model and can fail on worn
markings, shadows, curves, intersections, and adverse weather.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneEstimate:
    status: str
    confidence: float
    corridor: tuple[tuple[float, float], ...] | None
    left_boundary: tuple[tuple[float, float], tuple[float, float]] | None
    right_boundary: tuple[tuple[float, float], tuple[float, float]] | None
    left_segments: int
    right_segments: int


class LaneDetector:
    """Estimate the ego-lane corridor from visible left/right lane markings."""

    def __init__(
        self,
        *,
        roi_top: float = 0.56,
        roi_bottom: float = 0.97,
        smoothing: float = 0.35,
        max_hold_frames: int = 8,
    ) -> None:
        if not 0.2 <= roi_top < roi_bottom <= 1.0:
            raise ValueError("ROI bounds must satisfy 0.2 <= top < bottom <= 1.0")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if max_hold_frames < 0:
            raise ValueError("max_hold_frames cannot be negative")

        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.smoothing = float(smoothing)
        self.max_hold_frames = int(max_hold_frames)
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self._missing_frames = 0
        self._previous_confidence = 0.0

    def reset(self) -> None:
        self._previous = None
        self._missing_frames = 0
        self._previous_confidence = 0.0

    @staticmethod
    def _fit_boundary(
        segments: list[tuple[int, int, int, int]], top_y: float, bottom_y: float
    ) -> tuple[float, float] | None:
        if not segments:
            return None
        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []
        for x1, y1, x2, y2 in segments:
            length = math.hypot(x2 - x1, y2 - y1)
            xs.extend((float(x1), float(x2)))
            ys.extend((float(y1), float(y2)))
            weights.extend((length, length))
        if max(ys) - min(ys) < 20:
            return None

        slope, intercept = np.polyfit(
            np.asarray(ys),
            np.asarray(xs),
            1,
            w=np.sqrt(np.asarray(weights)),
        )
        x_top = float(slope * top_y + intercept)
        x_bottom = float(slope * bottom_y + intercept)
        if not math.isfinite(x_top) or not math.isfinite(x_bottom):
            return None
        return x_top, x_bottom

    def _make_estimate(
        self,
        status: str,
        confidence: float,
        left: np.ndarray,
        right: np.ndarray,
        width: int,
        height: int,
        left_count: int,
        right_count: int,
    ) -> LaneEstimate:
        top_y = self.roi_top
        bottom_y = self.roi_bottom
        left_pixels = (
            (float(left[0]), top_y * height),
            (float(left[1]), bottom_y * height),
        )
        right_pixels = (
            (float(right[0]), top_y * height),
            (float(right[1]), bottom_y * height),
        )
        corridor = (
            (left_pixels[0][0] / width, top_y),
            (right_pixels[0][0] / width, top_y),
            (right_pixels[1][0] / width, bottom_y),
            (left_pixels[1][0] / width, bottom_y),
        )
        return LaneEstimate(
            status=status,
            confidence=max(0.0, min(1.0, confidence)),
            corridor=corridor,
            left_boundary=left_pixels,
            right_boundary=right_pixels,
            left_segments=left_count,
            right_segments=right_count,
        )

    def detect(self, frame: np.ndarray) -> LaneEstimate:
        """Estimate lane boundaries for a BGR frame."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image")
        height, width = frame.shape[:2]
        if height < 32 or width < 32:
            raise ValueError("frame dimensions must be at least 32x32")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 55, 150)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv, np.array((0, 0, 175), dtype=np.uint8),
            np.array((180, 75, 255), dtype=np.uint8),
        )
        yellow = cv2.inRange(
            hsv, np.array((12, 65, 75), dtype=np.uint8),
            np.array((42, 255, 255), dtype=np.uint8),
        )
        marking_edges = cv2.Canny(cv2.bitwise_or(white, yellow), 40, 120)
        combined = cv2.bitwise_or(edges, marking_edges)

        top_y = int(round(self.roi_top * height))
        bottom_y = int(round(self.roi_bottom * height))
        roi = np.zeros_like(combined)
        roi_polygon = np.array(
            [[
                (int(0.04 * width), bottom_y),
                (int(0.39 * width), top_y),
                (int(0.61 * width), top_y),
                (int(0.96 * width), bottom_y),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, roi_polygon, 255)
        masked = cv2.bitwise_and(combined, roi)

        threshold = max(18, int(width * 0.018))
        min_length = max(28, int(width * 0.035))
        max_gap = max(24, int(width * 0.08))
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        left_segments: list[tuple[int, int, int, int]] = []
        right_segments: list[tuple[int, int, int, int]] = []
        if lines is not None:
            for raw in lines[:, 0, :]:
                x1, y1, x2, y2 = (int(value) for value in raw)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                slope = dy / dx
                midpoint_x = (x1 + x2) * 0.5
                if 0.30 <= abs(slope) <= 3.5:
                    if slope < 0 and midpoint_x < width * 0.58:
                        left_segments.append((x1, y1, x2, y2))
                    elif slope > 0 and midpoint_x > width * 0.42:
                        right_segments.append((x1, y1, x2, y2))

        left_fit = self._fit_boundary(left_segments, top_y, bottom_y)
        right_fit = self._fit_boundary(right_segments, top_y, bottom_y)
        valid = left_fit is not None and right_fit is not None

        if valid:
            left = np.asarray(left_fit, dtype=np.float64)
            right = np.asarray(right_fit, dtype=np.float64)
            bottom_width = right[1] - left[1]
            top_width = right[0] - left[0]
            valid = (
                0.18 * width <= bottom_width <= 0.95 * width
                and 0.08 * width <= top_width <= 0.90 * width
                and left[0] < right[0]
                and left[1] < right[1]
                and min(left[0], left[1], right[0], right[1]) >= -0.05 * width
                and max(left[0], left[1], right[0], right[1]) <= 1.05 * width
            )
        if valid:
            if self._previous is not None:
                previous_left, previous_right = self._previous
                alpha = self.smoothing
                left = alpha * left + (1.0 - alpha) * previous_left
                right = alpha * right + (1.0 - alpha) * previous_right
            self._previous = (left.copy(), right.copy())
            self._missing_frames = 0
            confidence = min(
                0.98,
                0.40 + 0.055 * min(len(left_segments), 6)
                + 0.055 * min(len(right_segments), 6),
            )
            self._previous_confidence = confidence
            return self._make_estimate(
                "DETECTED", confidence, left, right, width, height,
                len(left_segments), len(right_segments),
            )

        self._missing_frames += 1
        if self._previous is not None and self._missing_frames <= self.max_hold_frames:
            confidence = self._previous_confidence * (
                1.0 - self._missing_frames / (self.max_hold_frames + 1)
            )
            left, right = self._previous
            return self._make_estimate(
                "TRACKED", confidence, left, right, width, height,
                len(left_segments), len(right_segments),
            )

        self._previous = None
        self._previous_confidence = 0.0
        return LaneEstimate(
            status="NOT_DETECTED",
            confidence=0.0,
            corridor=None,
            left_boundary=None,
            right_boundary=None,
            left_segments=len(left_segments),
            right_segments=len(right_segments),
        )
