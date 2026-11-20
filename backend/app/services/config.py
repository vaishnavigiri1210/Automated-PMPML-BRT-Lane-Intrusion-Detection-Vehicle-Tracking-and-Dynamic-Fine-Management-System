"""Configurable penalty, distance stage, and corridor monitoring policy."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PenaltyPolicy:
    min_consecutive_frames: int = 2          # Consecutive in-lane frames required to confirm intrusion
    exit_reset_frames: int = 2               # Consecutive out-of-lane frames before resetting streak
    meters_per_frame_estimate: float = 2.5   # Realistic fallback distance step per frame (meters)

    # Stage distance thresholds (meters) inside the dedicated BRT corridor
    stage1_distance_m: float = 250.0
    stage2_distance_m: float = 1000.0
    stage3_distance_m: float = 1500.0

    # Progressive fine amounts (INR)
    base_fine_stage1: int = 500
    base_fine_stage2: int = 1200
    base_fine_stage3: int = 2000

    # Stage 3 interest calculation parameters (INR)
    stage3_base_interest_inr: int = 100
    stage3_interest_per_meter: float = 0.2