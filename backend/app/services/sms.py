"""Stage-wise SMS alert service for PMPML BRT lane intrusion violations."""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Set, Tuple

logger = logging.getLogger("brts.sms")


@dataclass(frozen=True)
class StagePolicy:
    stage: int
    distance_m: float
    base_fine_inr: int
    has_interest: bool = False


STAGE_POLICIES: Dict[int, StagePolicy] = {
    0: StagePolicy(stage=0, distance_m=0.0, base_fine_inr=0, has_interest=False),
    1: StagePolicy(stage=1, distance_m=250.0, base_fine_inr=500, has_interest=False),
    2: StagePolicy(stage=2, distance_m=1000.0, base_fine_inr=1200, has_interest=False),
    3: StagePolicy(stage=3, distance_m=1500.0, base_fine_inr=2000, has_interest=True),
}


class SMSNotificationService:
    """Dispatch entry warning (Stage 0) and fine escalation alerts (Stages 1-3) with deduplication."""

    def __init__(self) -> None:
        # Track (track_id, stage) pairs already notified to prevent duplicate per-frame alerts
        self._sent_tracker: Set[Tuple[int, int]] = set()
        self.provider = os.getenv("SMS_PROVIDER", "SIMULATION").upper()
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_from = os.getenv("TWILIO_FROM_NUMBER", "")

    @staticmethod
    def build_message(
        plate_number: str,
        stage: int,
        distance_m: float,
        fine_inr: int,
        interest_inr: int = 0,
    ) -> str:
        """Build stage-specific warning and fine escalation SMS text."""
        if stage == 0:
            return (
                f"PMPML BRT ENTRY WARNING [{plate_number}]: Unauthorized entry detected in the "
                f"dedicated BRT lane. Please exit the lane immediately within 250m to avoid a Rs.500 fine."
            )
        if stage == 1:
            return (
                f"PMPML BRT FINE NOTICE [{plate_number}]: Continued inside dedicated BRT lane "
                f"(~{int(distance_m)}m). Stage 1 fine of Rs.{fine_inr} has been issued. "
                f"Exit immediately to prevent further escalation."
            )
        if stage == 2:
            return (
                f"PMPML BRT ESCALATION [{plate_number}]: Still inside BRT corridor (~{int(distance_m)}m). "
                f"Fine escalated to Stage 2 (Rs.{fine_inr}). "
                f"Exit or take a permissible turn immediately."
            )
        total_amount = fine_inr + interest_inr
        interest_note = (
            f" + Rs.{interest_inr} accrued interest (Total Rs.{total_amount})"
            if interest_inr > 0
            else " + applicable interest"
        )
        return (
            f"PMPML BRT CRITICAL [{plate_number}]: Continued BRT lane intrusion (~{int(distance_m)}m). "
            f"Escalated to Stage 3 (Rs.{fine_inr}{interest_note}). E-challan recorded."
        )

    def send_stage_alert(
        self,
        track_id: int,
        plate_number: str,
        owner_phone: str,
        stage: int,
        distance_m: float,
        fine_inr: int,
        interest_inr: int = 0,
    ) -> bool:
        """Send an SMS alert once per (track_id, stage) pair via Twilio or simulation log."""
        key = (track_id, stage)
        if key in self._sent_tracker:
            return False

        body = self.build_message(
            plate_number=plate_number,
            stage=stage,
            distance_m=distance_m,
            fine_inr=fine_inr,
            interest_inr=interest_inr,
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.provider == "TWILIO" and self.twilio_sid and self.twilio_token and self.twilio_from:
            from twilio.rest import Client  # type: ignore

            client = Client(self.twilio_sid, self.twilio_token)
            msg = client.messages.create(body=body, from_=self.twilio_from, to=owner_phone)
            logger.info("SMS sent via Twilio SID=%s to %s at %s", msg.sid, owner_phone, timestamp)
        else:
            logger.info(
                "[SMS SIMULATED | %s] To: %s | Stage: %d | Plate: %s | Msg: %s",
                timestamp,
                owner_phone,
                stage,
                plate_number,
                body,
            )

        self._sent_tracker.add(key)
        return True

    def reset_track(self, track_id: int) -> None:
        """Clear sent stage records for a specific track ID."""
        self._sent_tracker = {item for item in self._sent_tracker if item[0] != track_id}