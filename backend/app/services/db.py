"""Persistent PostgreSQL storage layer for PMPML BRTS violations and SMS logs."""
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("brts.db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:1234@127.0.0.1:5432/brts_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ViolationRecord(Base):
    """Stores confirmed and stage-escalated BRT lane intrusion events."""
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(32), unique=True, index=True, nullable=False)
    track_id = Column(Integer, index=True, nullable=False)
    category = Column(String(32), nullable=False)
    plate_number = Column(String(32), nullable=False)
    plate_confidence = Column(Float, default=0.0)
    timestamp = Column(String(32), nullable=False)
    distance_m = Column(Float, default=0.0)
    stage = Column(Integer, default=1)
    fine_amount = Column(Integer, default=500)
    interest_amount = Column(Integer, default=0)
    total_fine = Column(Integer, default=500)
    status = Column(String(64), default="CONFIRMED_VIOLATION")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SMSLogRecord(Base):
    """Stores dispatched Stage 1, 2, and 3 owner SMS notifications."""
    __tablename__ = "sms_logs"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True, nullable=False)
    plate_number = Column(String(32), nullable=False)
    owner_phone = Column(String(32), nullable=False)
    stage = Column(Integer, nullable=False)
    distance_m = Column(Float, default=0.0)
    fine_inr = Column(Integer, default=500)
    interest_inr = Column(Integer, default=0)
    status = Column(String(64), default="DELIVERED (SIMULATED)")
    message = Column(Text, nullable=False)
    timestamp = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def init_db() -> bool:
    """Create database tables if they do not already exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL tables initialized successfully in brts_db.")
        return True
    except Exception as exc:
        logger.warning("PostgreSQL connection not ready (%s). Running with in-memory fallback.", exc)
        return False


def save_violation_to_db(payload: Dict[str, Any]) -> None:
    """Insert a new violation record or update an existing one by event_id."""
    try:
        with SessionLocal() as db:
            existing = db.execute(
                select(ViolationRecord).where(ViolationRecord.event_id == payload["event_id"])
            ).scalar_one_or_none()

            if existing:
                existing.plate_number = payload["plate_number"]
                existing.plate_confidence = float(payload.get("plate_confidence", 0.0))
                existing.distance_m = float(payload.get("distance_m", existing.distance_m))
                existing.stage = int(payload.get("stage", existing.stage))
                existing.fine_amount = int(payload.get("fine_amount", existing.fine_amount))
                existing.interest_amount = int(payload.get("interest_amount", existing.interest_amount))
                existing.total_fine = int(payload.get("total_fine", existing.total_fine))
                existing.status = payload.get("status", existing.status)
            else:
                rec = ViolationRecord(
                    event_id=payload["event_id"],
                    track_id=int(payload["track_id"]),
                    category=payload["category"],
                    plate_number=payload["plate_number"],
                    plate_confidence=float(payload.get("plate_confidence", 0.0)),
                    timestamp=payload["timestamp"],
                    distance_m=float(payload.get("distance_m", 0.0)),
                    stage=int(payload.get("stage", 1)),
                    fine_amount=int(payload.get("fine_amount", 500)),
                    interest_amount=int(payload.get("interest_amount", 0)),
                    total_fine=int(payload.get("total_fine", 500)),
                    status=payload.get("status", "CONFIRMED_VIOLATION"),
                )
                db.add(rec)
            db.commit()
    except Exception as exc:
        logger.debug("Skipped DB violation persist: %s", exc)


def save_sms_log_to_db(entry: Dict[str, Any]) -> None:
    """Persist a dispatched SMS alert record in PostgreSQL."""
    try:
        with SessionLocal() as db:
            rec = SMSLogRecord(
                track_id=int(entry["track_id"]),
                plate_number=entry["plate_number"],
                owner_phone=entry["owner_phone"],
                stage=int(entry["stage"]),
                distance_m=float(entry.get("distance_m", 0.0)),
                fine_inr=int(entry.get("fine_inr", 500)),
                interest_inr=int(entry.get("interest_inr", 0)),
                status=entry.get("status", "DELIVERED (SIMULATED)"),
                message=entry["message"],
                timestamp=entry["timestamp"],
            )
            db.add(rec)
            db.commit()
    except Exception as exc:
        logger.debug("Skipped DB SMS log persist: %s", exc)


def load_recent_violations_from_db(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch latest violation records from PostgreSQL."""
    try:
        with SessionLocal() as db:
            rows = db.execute(
                select(ViolationRecord).order_by(ViolationRecord.id.desc()).limit(limit)
            ).scalars().all()
            return [
                {
                    "event_id": r.event_id,
                    "track_id": r.track_id,
                    "category": r.category,
                    "plate_number": r.plate_number,
                    "plate_confidence": r.plate_confidence,
                    "timestamp": r.timestamp,
                    "distance_m": r.distance_m,
                    "stage": r.stage,
                    "fine_amount": r.fine_amount,
                    "interest_amount": r.interest_amount,
                    "total_fine": r.total_fine,
                    "status": r.status,
                }
                for r in rows
            ]
    except Exception:
        return []


def load_total_violations_count_from_db() -> int:
    """Return total violation count stored in PostgreSQL."""
    try:
        with SessionLocal() as db:
            count = db.execute(select(func.count(ViolationRecord.id))).scalar()
            return int(count or 0)
    except Exception:
        return 0


def load_total_sms_count_from_db() -> int:
    """Return total SMS alert count stored in PostgreSQL."""
    try:
        with SessionLocal() as db:
            count = db.execute(select(func.count(SMSLogRecord.id))).scalar()
            return int(count or 0)
    except Exception:
        return 0


def load_recent_sms_from_db(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch latest SMS logs from PostgreSQL."""
    try:
        with SessionLocal() as db:
            rows = db.execute(
                select(SMSLogRecord).order_by(SMSLogRecord.id.desc()).limit(limit)
            ).scalars().all()
            return [
                {
                    "timestamp": r.timestamp,
                    "track_id": r.track_id,
                    "plate_number": r.plate_number,
                    "owner_phone": r.owner_phone,
                    "stage": r.stage,
                    "distance_m": r.distance_m,
                    "fine_inr": r.fine_inr,
                    "interest_inr": r.interest_inr,
                    "status": r.status,
                    "message": r.message,
                }
                for r in rows
            ]
    except Exception:
        return []