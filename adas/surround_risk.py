"""Camera-local occupancy and apparent-approach warnings for surround views.

Regions are normalized image ROIs, not calibrated vehicle-space blind spots.
No lane-change permission, metric distance, or vehicle command is produced.
"""

from dataclasses import replace
from typing import Iterable, Sequence

from adas.risk_assessment import CAUTION, CLEAR, CollisionRiskAssessor, FrameRisk

SURROUND_POSITIONS = ("left", "right", "rear_left", "rear_right")
# Identical defaults intentionally avoid assuming mirrored or asymmetric optics.
# Adjust each ROI to the mounting angle using --warning-roi.
DEFAULT_WARNING_ROI = ((0.05, 0.30), (0.95, 0.30), (0.95, 1.0), (0.05, 1.0))


class SurroundRiskAssessor:
    """Warn about occupied side/rear regions, escalating for apparent approach.

    A stationary-size track still produces CAUTION inside the configured region.
    Expansion TTC uses the existing estimator, but its front cut-in projection is
    deliberately not used for side/rear cameras. Each instance owns its history.
    """

    def __init__(
        self,
        position: str,
        *,
        warning_roi: Sequence[Sequence[float]] | None = None,
        pedestrian_class_ids: Iterable[int] = (0,),
        vehicle_class_ids: Iterable[int] = (1, 2, 3, 5, 7),
    ) -> None:
        if position not in SURROUND_POSITIONS:
            raise ValueError(f"Unknown surround camera position: {position}")
        self.position = position
        roi = DEFAULT_WARNING_ROI if warning_roi is None else warning_roi
        if len(roi) < 3:
            raise ValueError("warning ROI needs at least three normalized points")
        self.estimator = CollisionRiskAssessor(
            ego_corridor=roi,
            pedestrian_class_ids=pedestrian_class_ids,
            vehicle_class_ids=vehicle_class_ids,
        )

    def reset(self) -> None:
        self.estimator.reset()

    def update(
        self,
        tracked_objects: Iterable[Sequence[float]],
        frame_shape: tuple[int, ...],
        *,
        timestamp: float | None = None,
    ) -> FrameRisk:
        base = self.estimator.update(tracked_objects, frame_shape, timestamp=timestamp)
        objects = []
        for item in base.objects:
            eligible = item.in_ego_corridor and item.object_type != "other"
            if eligible:
                level = self.estimator._level_for_ttc(item.ttc_seconds)
                level = CAUTION if level == CLEAR else level
                kind = "Side occupancy" if self.position in {"left", "right"} else "Rear occupancy"
                reason = f"{self.position}: {kind}, {item.object_type} track {item.track_id}"
                if item.ttc_seconds is not None:
                    reason += f"; apparent-expansion TTC ~{item.ttc_seconds:.1f}s"
            else:
                level, reason = CLEAR, "Outside warning ROI or non-obstacle class"
            objects.append(replace(item, level=level, reason=reason, cut_in_eta_seconds=None))
        rank = {"CLEAR": 0, "CAUTION": 1, "HIGH": 2, "CRITICAL": 3}
        highest = max(objects, key=lambda item: rank[item.level], default=None)
        return FrameRisk(
            highest.level if highest else CLEAR,
            highest.reason if highest and highest.level != CLEAR else "No tracked obstacle in warning ROI",
            tuple(objects), base.corridor,
        )
