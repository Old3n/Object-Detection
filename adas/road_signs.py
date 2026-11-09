"""Iranian road-sign readings and low-light frame enhancement.

Sign readings come from detector class names only (COCO "stop sign", the ITSD
class codes, or custom "speed_<km/h>" / "junction_<type>" classes). Nothing
here reads sign text, measures distance, or produces a vehicle command.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Iterable

import cv2
import numpy as np

SignReading = tuple[str, int | None]

STOP_NAMES = {"stop sign", "stop"}
SPEED_NAME = re.compile(r"^speed_(\d+)$")
SPEED_MIN_KMH, SPEED_MAX_KMH = 10, 50
# ITSD class codes (data_labels.yaml), checked against sample images from Fold-1.
# d-r-20 is a police inspection stop and d-r-speed mixes 15, 85 and 95, so
# neither is listed. r-g-plate mixes route plates with other signs.
ITSD_SIGNS: dict[str, SignReading] = {
    "d-xx-1": ("stop", None),
    "d-r-30": ("speed", 30),
    "d-r-31": ("speed", 40),
    "d-r-32": ("speed", 50),
    "d-r-33": ("speed", 60),
    "d-r-34": ("speed", 70),
    "d-r-35": ("speed", 80),
    "d-r-36": ("speed", 90),
    "d-r-37": ("speed", 100),
    "d-r-38": ("speed", 110),
    "d-r-39": ("speed", 120),
    "w-r- intersection": ("junction", None),
    "r-g-k": ("route", None),  # logged only; destinations are not read
}
CONFIRM_FRAMES = 3
NIGHT_BRIGHTNESS = 60  # mean pixel value (0-255) below which a frame counts as night


def sign_kind(name: str) -> SignReading | None:
    """Return (kind, value) for a target sign class, or None for any other class."""
    match = SPEED_NAME.match(name)
    if name in ITSD_SIGNS:
        kind, value = ITSD_SIGNS[name]
    elif name in STOP_NAMES:
        kind, value = "stop", None
    elif match:
        kind, value = "speed", int(match.group(1))
    elif name.startswith("junction_"):
        kind, value = "junction", None
    else:
        return None
    if kind == "speed" and not SPEED_MIN_KMH <= value <= SPEED_MAX_KMH:
        return None
    return kind, value


class SignConfirmer:
    """Accept a reading only when it appears in `frames` consecutive frames."""

    def __init__(self, frames: int = CONFIRM_FRAMES) -> None:
        if frames < 1:
            raise ValueError("frames must be at least 1")
        self.history: deque[set[SignReading]] = deque(maxlen=frames)

    def reset(self) -> None:
        self.history.clear()

    def update(self, readings: Iterable[SignReading]) -> set[SignReading]:
        self.history.append(set(readings))
        if len(self.history) < self.history.maxlen:
            return set()
        return set.intersection(*self.history)


def sign_text(confirmed: Iterable[SignReading]) -> str:
    """One-line summary of confirmed signs for the HUD."""
    confirmed = set(confirmed)
    parts = [label for kind, label in (("stop", "STOP"), ("junction", "JUNCTION"), ("route", "ROUTE"))
             if (kind, None) in confirmed]
    parts += [f"SPEED {value}" for value in sorted(v for k, v in confirmed if k == "speed")]
    return " | ".join(parts) if parts else "Signs: none"


def is_night(frame: np.ndarray, threshold: float = NIGHT_BRIGHTNESS) -> bool:
    return float(frame.mean()) < threshold


def boost_night(frame: np.ndarray) -> np.ndarray:
    """Raise local contrast with CLAHE on the LAB lightness channel."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(light), green_red, blue_yellow))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
