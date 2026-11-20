"""Intrusion event management, entry tracking, and distance-based fine calculation."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

from app.services.config import PenaltyPolicy


@dataclass
class ActiveIntrusion:
    """Represents a vehicle currently inside the BRT lane (even before reaching 250m fine threshold)."""
    track_id: int
    category: str
    plate_number: str
    plate_confidence: float
    distance_m: float


@dataclass
class LiveViolation:
    """Represents a confirmed fine-bearing violation once distance >= 250m."""
    event_id: str
    track_id: int
    category: str
    plate_number: str
    timestamp: str
    distance_m: float
    stage: int
    fine_amount: int
    interest_amount: int = 0
    plate_confidence: float = 0.0
    status: str = "CONFIRMED_VIOLATION"
    evidence_path: Optional[str] = None

    @property
    def total_fine(self) -> int:
        """Return total payable penalty including any accrued Stage 3 interest."""
        return self.fine_amount + self.interest_amount


class RealtimeTracker:
    """Tracks BRT lane entries, emits entry telemetry at 0m, and enforces fines starting at 250m."""

    def __init__(self, min_frames: int = 2, policy: Optional[PenaltyPolicy] = None):
        self.policy = policy or PenaltyPolicy(min_consecutive_frames=min_frames)
        self.min_frames = min_frames
        self.in_lane_counts: Dict[int, int] = {}
        self.out_lane_counts: Dict[int, int] = {}
        self.distance_traveled_m: Dict[int, float] = {}
        self.track_plates: Dict[int, str] = {}
        self.track_plate_conf: Dict[int, float] = {}
        self.violations: List[LiveViolation] = []
        self.violations_by_track: Dict[int, LiveViolation] = {}
        self.seen_violators: set = set()

    def _compute_stage_and_fine(self, distance_m: float) -> Optional[Tuple[int, int, int]]:
        """Return (stage, base_fine_inr, interest_inr) only after reaching Stage 1 distance (250m)."""
        if distance_m < self.policy.stage1_distance_m:
            return None

        if distance_m >= self.policy.stage3_distance_m:
            extra_meters = max(0.0, distance_m - self.policy.stage3_distance_m)
            interest = int(
                self.policy.stage3_base_interest_inr
                + extra_meters * self.policy.stage3_interest_per_meter
            )
            return 3, self.policy.base_fine_stage3, interest

        if distance_m >= self.policy.stage2_distance_m:
            return 2, self.policy.base_fine_stage2, 0

        return 1, self.policy.base_fine_stage1, 0

    def update(
        self,
        track_id: int,
        category: str,
        in_lane: bool,
        detected_plate: Optional[str] = None,
        plate_conf: float = 0.0,
        step_distance_m: Optional[float] = None,
    ) -> Tuple[Optional[ActiveIntrusion], Optional[LiveViolation]]:
        """Update lane state and return (active_intrusion, fine_violation_if_250m_crossed)."""
        if detected_plate:
            cleaned_plate = detected_plate.strip().upper()
            if cleaned_plate and cleaned_plate != "UNREADABLE":
                if plate_conf >= self.track_plate_conf.get(track_id, 0.0):
                    self.track_plates[track_id] = cleaned_plate
                    self.track_plate_conf[track_id] = plate_conf

        if not in_lane:
            self.out_lane_counts[track_id] = self.out_lane_counts.get(track_id, 0) + 1
            if self.out_lane_counts[track_id] >= self.policy.exit_reset_frames:
                self.in_lane_counts[track_id] = 0
            return None, None

        self.out_lane_counts[track_id] = 0
        self.in_lane_counts[track_id] = self.in_lane_counts.get(track_id, 0) + 1

        delta_m = (
            float(step_distance_m)
            if step_distance_m is not None and step_distance_m >= 0.0
            else self.policy.meters_per_frame_estimate
        )
        updated_distance = self.distance_traveled_m.get(track_id, 0.0) + delta_m
        self.distance_traveled_m[track_id] = updated_distance

        if self.in_lane_counts[track_id] < self.min_frames:
            return None, None

        plate = self.track_plates.get(track_id, "UNREADABLE")
        conf_val = self.track_plate_conf.get(track_id, 0.0)

        active = ActiveIntrusion(
            track_id=track_id,
            category=category.upper(),
            plate_number=plate,
            plate_confidence=round(conf_val, 2),
            distance_m=round(updated_distance, 1),
        )

        stage_info = self._compute_stage_and_fine(updated_distance)
        if stage_info is None:
            return active, None

        stage, fine_inr, interest_inr = stage_info

        if track_id not in self.seen_violators:
            self.seen_violators.add(track_id)
            violation = LiveViolation(
                event_id=str(uuid.uuid4())[:8].upper(),
                track_id=track_id,
                category=category.upper(),
                plate_number=plate,
                timestamp=datetime.now().strftime("%H:%M:%S"),
                distance_m=round(updated_distance, 1),
                stage=stage,
                fine_amount=fine_inr,
                interest_amount=interest_inr,
                plate_confidence=round(conf_val, 2),
                status="CONFIRMED_VIOLATION",
            )
            self.violations_by_track[track_id] = violation
            self.violations.insert(0, violation)
            return active, violation

        existing = self.violations_by_track.get(track_id)
        if existing is not None:
            existing.distance_m = round(updated_distance, 1)
            if plate != "UNREADABLE":
                existing.plate_number = plate
            existing.plate_confidence = round(conf_val, 2)
            if stage > existing.stage or (stage == 3 and interest_inr != existing.interest_amount):
                existing.stage = stage
                existing.fine_amount = fine_inr
                existing.interest_amount = interest_inr
                existing.status = f"ESCALATED_STAGE_{stage}"
            return active, existing

        return active, None