"""Monocular, track-based collision-risk estimation for ADAS research.

TTC is estimated from tracked bounding-box expansion. This is advisory perception
logic, not a metric distance sensor or a vehicle controller. The normalized
ego-corridor polygon is a configurable image ROI, not a detected lane boundary.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

CLEAR = "CLEAR"
CAUTION = "CAUTION"
HIGH = "HIGH"
CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ObjectRisk:
    track_id: int
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    object_type: str
    in_ego_corridor: bool
    ttc_seconds: float | None
    lateral_velocity_norm_s: float | None
    cut_in_eta_seconds: float | None
    level: str
    reason: str


@dataclass(frozen=True)
class FrameRisk:
    level: str
    reason: str
    objects: tuple[ObjectRisk, ...]
    corridor: tuple[tuple[float, float], ...]


class CollisionRiskAssessor:
    """Estimate risk using apparent expansion and lateral track motion.

    Apparent-expansion TTC assumes object dimensions and camera geometry remain
    approximately stable while radial expansion dominates. It is not metric range.
    """

    DEFAULT_CORRIDOR = (
        (0.30, 0.40),
        (0.70, 0.40),
        (0.95, 0.95),
        (0.05, 0.95),
    )

    def __init__(
        self,
        *,
        ego_corridor: Sequence[Sequence[float]] | None = None,
        pedestrian_class_ids: Iterable[int] = (0,),
        vehicle_class_ids: Iterable[int] = (1, 2, 3, 5, 7),
        min_confidence: float = 0.20,
        min_samples: int = 5,
        min_history_seconds: float = 0.30,
        stale_track_seconds: float = 1.0,
        critical_ttc_seconds: float = 1.5,
        high_ttc_seconds: float = 3.0,
        caution_ttc_seconds: float = 5.0,
        cut_in_horizon_seconds: float = 1.0,
    ) -> None:
        corridor = ego_corridor or self.DEFAULT_CORRIDOR
        if len(corridor) < 3:
            raise ValueError("ego_corridor must contain at least three points")
        self.ego_corridor = tuple((float(x), float(y)) for x, y in corridor)
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in self.ego_corridor):
            raise ValueError("ego_corridor coordinates must be normalized to 0..1")
        if min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range 0..1")
        if not (0 < critical_ttc_seconds < high_ttc_seconds < caution_ttc_seconds):
            raise ValueError("TTC thresholds must satisfy 0 < critical < high < caution")

        self.pedestrian_class_ids = set(pedestrian_class_ids)
        self.vehicle_class_ids = set(vehicle_class_ids)
        self.min_confidence = float(min_confidence)
        self.min_samples = int(min_samples)
        self.min_history_seconds = float(min_history_seconds)
        self.stale_track_seconds = float(stale_track_seconds)
        self.critical_ttc_seconds = float(critical_ttc_seconds)
        self.high_ttc_seconds = float(high_ttc_seconds)
        self.caution_ttc_seconds = float(caution_ttc_seconds)
        self.cut_in_horizon_seconds = float(cut_in_horizon_seconds)
        self.history: dict[int, deque[tuple[float, float, float, float]]] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self.last_seen: dict[int, float] = {}

    def reset(self) -> None:
        self.history.clear()
        self.last_seen.clear()

    @staticmethod
    def point_in_polygon(
        point: tuple[float, float], polygon: Sequence[Sequence[float]]
    ) -> bool:
        """Return whether a point lies inside a polygon using ray casting."""
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
            on_edge = (
                abs(cross) < 1e-10
                and min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10
                and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10
            )
            if on_edge:
                return True
            if (yi > y) != (yj > y):
                boundary_x = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < boundary_x:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _linear_slope(samples: Sequence[tuple[float, float]]) -> float | None:
        if len(samples) < 2:
            return None
        t0 = samples[0][0]
        times = [sample[0] - t0 for sample in samples]
        values = [sample[1] for sample in samples]
        mean_t = sum(times) / len(times)
        mean_v = sum(values) / len(values)
        denominator = sum((t - mean_t) ** 2 for t in times)
        if denominator <= 1e-12:
            return None
        return sum(
            (t - mean_t) * (value - mean_v)
            for t, value in zip(times, values)
        ) / denominator

    def _ttc(
        self, samples: Sequence[tuple[float, float, float, float]]
    ) -> float | None:
        if len(samples) < self.min_samples:
            return None
        if samples[-1][0] - samples[0][0] < self.min_history_seconds:
            return None
        log_widths = [
            (sample[0], math.log(sample[1]))
            for sample in samples
            if sample[1] > 0
        ]
        slope = self._linear_slope(log_widths)
        if slope is None or slope <= 0.02:
            return None
        ttc = 1.0 / slope
        return ttc if math.isfinite(ttc) and ttc > 0 else None

    def _level_for_ttc(self, ttc: float | None) -> str:
        if ttc is None or ttc > self.caution_ttc_seconds:
            return CLEAR
        if ttc <= self.critical_ttc_seconds:
            return CRITICAL
        if ttc <= self.high_ttc_seconds:
            return HIGH
        return CAUTION

    def update(
        self,
        tracked_objects: Iterable[Sequence[float]],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        *,
        timestamp: float | None = None,
        pedestrian_class_ids: Iterable[int] | None = None,
    ) -> FrameRisk:
        """Accept rows of x1,y1,x2,y2,track_id,confidence,class_id."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame_shape must have positive height and width")
        now = time.monotonic() if timestamp is None else float(timestamp)
        pedestrian_ids = (
            self.pedestrian_class_ids
            if pedestrian_class_ids is None
            else set(pedestrian_class_ids)
        )
        results: list[ObjectRisk] = []

        for raw in tracked_objects:
            if len(raw) < 7:
                continue
            x1, y1, x2, y2 = (float(value) for value in raw[:4])
            track_id, confidence, class_id = int(raw[4]), float(raw[5]), int(raw[6])
            if confidence < self.min_confidence or not all(
                math.isfinite(value) for value in (x1, y1, x2, y2, confidence)
            ):
                continue
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 1.0 or box_height <= 1.0:
                continue

            previous_seen = self.last_seen.get(track_id)
            if previous_seen is not None and now - previous_seen > self.stale_track_seconds:
                self.history.pop(track_id, None)
            self.last_seen[track_id] = now

            center_x_norm = ((x1 + x2) * 0.5) / width
            bottom_y_norm = y2 / height
            width_norm = box_width / width
            history = self.history[track_id]
            history.append((now, width_norm, center_x_norm, bottom_y_norm))
            samples = tuple(history)

            in_corridor = self.point_in_polygon(
                (center_x_norm, bottom_y_norm), self.ego_corridor
            )
            ttc = self._ttc(samples)
            lateral_velocity = (
                self._linear_slope([(sample[0], sample[2]) for sample in samples])
                if len(samples) >= self.min_samples
                else None
            )

            cut_in_eta = None
            if (
                not in_corridor
                and lateral_velocity is not None
                and abs(lateral_velocity) > 0.01
            ):
                for step in range(1, 11):
                    eta = self.cut_in_horizon_seconds * step / 10
                    projected_x = center_x_norm + lateral_velocity * eta
                    if self.point_in_polygon(
                        (projected_x, bottom_y_norm), self.ego_corridor
                    ):
                        cut_in_eta = eta
                        break

            if class_id in pedestrian_ids:
                object_type = "pedestrian"
            elif class_id in self.vehicle_class_ids:
                object_type = "vehicle"
            else:
                object_type = "other"

            level = (
                self._level_for_ttc(ttc)
                if in_corridor and object_type != "other"
                else CLEAR
            )
            reason = ""
            if level != CLEAR:
                reason = (
                    f"{object_type.title()} track {track_id}: "
                    f"apparent-expansion TTC {ttc:.1f}s"
                )
            elif cut_in_eta is not None and object_type != "other":
                level = CAUTION
                reason = (
                    f"{object_type.title()} track {track_id}: "
                    f"predicted corridor entry in {cut_in_eta:.1f}s"
                )

            results.append(
                ObjectRisk(
                    track_id=track_id,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    object_type=object_type,
                    in_ego_corridor=in_corridor,
                    ttc_seconds=ttc,
                    lateral_velocity_norm_s=lateral_velocity,
                    cut_in_eta_seconds=cut_in_eta,
                    level=level,
                    reason=reason,
                )
            )

        expired = [
            track_id
            for track_id, seen_at in self.last_seen.items()
            if now - seen_at > self.stale_track_seconds
        ]
        for track_id in expired:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)

        if not results:
            return FrameRisk(CLEAR, "No eligible tracked obstacles", (), self.ego_corridor)
        rank = {CLEAR: 0, CAUTION: 1, HIGH: 2, CRITICAL: 3}
        highest = max(results, key=lambda item: rank[item.level])
        reason = (
            highest.reason
            if highest.level != CLEAR
            else "No elevated collision risk estimated"
        )
        return FrameRisk(highest.level, reason, tuple(results), self.ego_corridor)
