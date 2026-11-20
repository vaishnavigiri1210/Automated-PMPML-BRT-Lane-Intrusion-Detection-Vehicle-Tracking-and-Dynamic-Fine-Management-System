"""FastAPI application: dynamic video upload, real-time AI inference, and live streaming."""
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

from app.services.anpr import PlateReader
from app.services.authorization import AuthStatus, classify_authorization
from app.services.challan import generate_challan_pdf_bytes
from app.services.config import PenaltyPolicy
from app.services.db import (
    init_db,
    load_recent_sms_from_db,
    load_recent_violations_from_db,
    load_total_sms_count_from_db,
    load_total_violations_count_from_db,
    save_sms_log_to_db,
    save_violation_to_db,
)
from app.services.intrusion import RealtimeTracker
from app.services.lane import Point, estimate_step_distance_m, ground_point, point_in_polygon
from app.services.sms import SMSNotificationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("brts.main")

app = FastAPI(title="Pune BRTS Real-Time Surveillance Engine")

# Setup template, upload, and evidence snapshot directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
EVIDENCE_DIR = BASE_DIR / "evidence"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize PostgreSQL tables on startup
DB_AVAILABLE = init_db()

# Initialize YOLO model, penalty policy, tracker, ANPR reader, and SMS service
penalty_policy = PenaltyPolicy()
model = YOLO("yolov8n.pt")
plate_reader = PlateReader(min_confidence=0.35, min_ocr_interval=0.6)
tracker_state = RealtimeTracker(min_frames=penalty_policy.min_consecutive_frames, policy=penalty_policy)
sms_service = SMSNotificationService()
recent_sms_logs: List[dict] = load_recent_sms_from_db(limit=20) if DB_AVAILABLE else []
last_ground_points: Dict[int, Point] = {}
last_ocr_timestamps: Dict[int, float] = {}
evidence_by_event: Dict[str, str] = {}

MOCK_OWNER_PHONES: Dict[str, str] = {
    "MH12AB1234": "+919823011111",
    "MH14CD5678": "+919823022222",
    "MH12EF9012": "+919823033333",
    "MH14GH3456": "+919823044444",
    "MH12IJ7890": "+919823055555",
}

default_video = PROJECT_ROOT / "video_1.mp4"
CURRENT_VIDEO_SOURCE: Union[str, int] = str(default_video) if default_video.exists() else 0
STREAM_SESSION_ID: int = 1


class TestSMSRequest(BaseModel):
    track_id: int = 101
    plate_number: str = "MH12AB1234"
    stage: int = 0


def resolve_owner_phone(plate_number: str) -> str:
    """Resolve owner mobile number from normalized vehicle registration plate."""
    cleaned = plate_number.replace(" ", "").upper()
    return MOCK_OWNER_PHONES.get(cleaned, "+919800000000")


def run_throttled_ocr(
    frame: np.ndarray,
    track_id: int,
    bbox: Tuple[float, float, float, float],
) -> Tuple[Optional[str], float]:
    """Run EasyOCR periodically per track ID while a plate has not yet been confirmed."""
    if track_id in tracker_state.track_plates:
        return tracker_state.track_plates[track_id], tracker_state.track_plate_conf.get(track_id, 0.0)

    now = time.monotonic()
    last_time = last_ocr_timestamps.get(track_id, 0.0)
    if (now - last_time) < plate_reader.min_ocr_interval:
        return None, 0.0

    last_ocr_timestamps[track_id] = now
    return plate_reader.read_plate(frame=frame, bbox=bbox)


def capture_evidence_snapshot(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    event_id: str,
    plate_number: str,
    distance_m: float,
    stage: int,
) -> Optional[str]:
    """Save an annotated evidence JPEG for a confirmed violation and return its public URL."""
    try:
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"CHALLAN #{event_id} | {plate_number} | Stage {stage} ({distance_m:.0f}m) | {stamp}"
        cv2.putText(
            annotated,
            caption,
            (12, max(24, annotated.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        filename = f"{event_id}_stage{stage}.jpg"
        out_path = EVIDENCE_DIR / filename
        cv2.imwrite(str(out_path), annotated)
        url_path = f"/evidence/{filename}"
        evidence_by_event[event_id] = url_path
        return url_path
    except Exception as exc:
        logger.warning("Evidence snapshot capture failed for %s: %s", event_id, exc)
        return None


def record_sms_log(
    track_id: int,
    plate_number: str,
    owner_phone: str,
    stage: int,
    distance_m: float,
    fine_inr: int,
    interest_inr: int,
    status: str,
) -> dict:
    """Append a structured SMS dispatch entry in memory and persist to PostgreSQL."""
    msg_text = sms_service.build_message(
        plate_number=plate_number,
        stage=stage,
        distance_m=distance_m,
        fine_inr=fine_inr,
        interest_inr=interest_inr,
    )
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "track_id": track_id,
        "plate_number": plate_number,
        "owner_phone": owner_phone,
        "stage": stage,
        "distance_m": round(distance_m, 1),
        "fine_inr": fine_inr,
        "interest_inr": interest_inr,
        "status": status,
        "message": msg_text,
    }
    recent_sms_logs.insert(0, entry)
    del recent_sms_logs[50:]
    save_sms_log_to_db(entry)
    return entry


def dispatch_entry_warning_sms(
    track_id: int,
    plate_number: str,
    distance_m: float,
) -> None:
    """Send immediate exit warning SMS (Stage 0, Rs.0 fine) upon lane entry and plate detection."""
    if not plate_number or plate_number.strip().upper() == "UNREADABLE":
        return

    owner_phone = resolve_owner_phone(plate_number)
    sent = sms_service.send_stage_alert(
        track_id=track_id,
        plate_number=plate_number,
        owner_phone=owner_phone,
        stage=0,
        distance_m=distance_m,
        fine_inr=0,
        interest_inr=0,
    )
    if sent:
        status = "ENTRY WARNING (TWILIO)" if sms_service.provider == "TWILIO" else "ENTRY WARNING SENT (SIMULATED)"
        record_sms_log(
            track_id=track_id,
            plate_number=plate_number,
            owner_phone=owner_phone,
            stage=0,
            distance_m=distance_m,
            fine_inr=0,
            interest_inr=0,
            status=status,
        )


def dispatch_violation_stage_sms(
    track_id: int,
    plate_number: str,
    distance_m: float,
    stage: int,
    fine_inr: int,
    interest_inr: int,
) -> None:
    """Dispatch stage-wise fine SMS (Stages 1-3) only after distance >= 250m and valid plate OCR."""
    if not plate_number or plate_number.strip().upper() == "UNREADABLE":
        return

    if distance_m < penalty_policy.stage1_distance_m:
        return

    owner_phone = resolve_owner_phone(plate_number)
    sent = sms_service.send_stage_alert(
        track_id=track_id,
        plate_number=plate_number,
        owner_phone=owner_phone,
        stage=stage,
        distance_m=distance_m,
        fine_inr=fine_inr,
        interest_inr=interest_inr,
    )
    if sent:
        status = "FINE SMS SENT (TWILIO)" if sms_service.provider == "TWILIO" else "FINE SMS DELIVERED (SIMULATED)"
        record_sms_log(
            track_id=track_id,
            plate_number=plate_number,
            owner_phone=owner_phone,
            stage=stage,
            distance_m=distance_m,
            fine_inr=fine_inr,
            interest_inr=interest_inr,
            status=status,
        )


def reset_session_state(new_source: Union[str, int]) -> None:
    """Reset active stream tracker state when switching video sources while preserving DB history."""
    global CURRENT_VIDEO_SOURCE, STREAM_SESSION_ID, tracker_state, sms_service, last_ground_points, last_ocr_timestamps
    CURRENT_VIDEO_SOURCE = new_source
    STREAM_SESSION_ID += 1
    tracker_state = RealtimeTracker(min_frames=penalty_policy.min_consecutive_frames, policy=penalty_policy)
    sms_service = SMSNotificationService()
    last_ground_points.clear()
    last_ocr_timestamps.clear()


def generate_live_stream() -> Generator[bytes, None, None]:
    """Read frames from the active video source, run inference, and stream MJPEG."""
    global CURRENT_VIDEO_SOURCE, STREAM_SESSION_ID
    local_session = STREAM_SESSION_ID

    cap = cv2.VideoCapture(CURRENT_VIDEO_SOURCE)

    while True:
        if local_session != STREAM_SESSION_ID:
            break

        ret, frame = cap.read()
        if not ret:
            if isinstance(CURRENT_VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        h, w = frame.shape[:2]

        brt_polygon = [
            (int(w * 0.15), int(h * 0.95)),
            (int(w * 0.35), int(h * 0.35)),
            (int(w * 0.65), int(h * 0.35)),
            (int(w * 0.85), int(h * 0.95)),
        ]

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            verbose=False,
        )

        poly_pts = np.array(brt_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poly_pts], isClosed=True, color=(255, 140, 0), thickness=2)
        cv2.putText(
            frame,
            "BRT DEDICATED LANE",
            (int(w * 0.2), int(h * 0.4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 0),
            2,
        )

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            names = results[0].names

            for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                tid = int(track_id)
                category = names[cls_id]
                bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                pt = ground_point(bbox)

                prev_pt: Optional[Point] = last_ground_points.get(tid)
                step_m = estimate_step_distance_m(
                    prev_point=prev_pt,
                    curr_point=pt,
                    frame_height=h,
                    fallback_step_m=penalty_policy.meters_per_frame_estimate,
                )
                last_ground_points[tid] = pt

                in_lane = point_in_polygon(pt, brt_polygon)
                auth = classify_authorization(category, float(conf), min_conf=0.30)

                ocr_plate: Optional[str] = None
                ocr_conf: float = 0.0

                if auth == AuthStatus.PRIVATE:
                    if in_lane:
                        ocr_plate, ocr_conf = run_throttled_ocr(frame=frame, track_id=tid, bbox=bbox)

                    active_intrusion, violation = tracker_state.update(
                        track_id=tid,
                        category=category,
                        in_lane=in_lane,
                        detected_plate=ocr_plate,
                        plate_conf=ocr_conf,
                        step_distance_m=step_m,
                    )

                    if in_lane and active_intrusion is not None:
                        dispatch_entry_warning_sms(
                            track_id=tid,
                            plate_number=active_intrusion.plate_number,
                            distance_m=active_intrusion.distance_m,
                        )

                    if in_lane and violation is not None:
                        if not violation.evidence_path:
                            snap_url = capture_evidence_snapshot(
                                frame=frame,
                                bbox=bbox,
                                event_id=violation.event_id,
                                plate_number=violation.plate_number,
                                distance_m=violation.distance_m,
                                stage=violation.stage,
                            )
                            if snap_url:
                                violation.evidence_path = snap_url

                        save_violation_to_db(
                            {
                                "event_id": violation.event_id,
                                "track_id": violation.track_id,
                                "category": violation.category,
                                "plate_number": violation.plate_number,
                                "plate_confidence": violation.plate_confidence,
                                "timestamp": violation.timestamp,
                                "distance_m": violation.distance_m,
                                "stage": violation.stage,
                                "fine_amount": violation.fine_amount,
                                "interest_amount": violation.interest_amount,
                                "total_fine": violation.total_fine,
                                "status": violation.status,
                            }
                        )
                        dispatch_violation_stage_sms(
                            track_id=tid,
                            plate_number=violation.plate_number,
                            distance_m=violation.distance_m,
                            stage=violation.stage,
                            fine_inr=violation.fine_amount,
                            interest_inr=violation.interest_amount,
                        )

                plate_tag = tracker_state.track_plates.get(tid)
                dist_now = tracker_state.distance_traveled_m.get(tid, 0.0)
                if in_lane and auth == AuthStatus.PRIVATE:
                    if dist_now >= penalty_policy.stage1_distance_m:
                        box_color = (0, 0, 255)
                        label = f"! FINE ! ID:{tid} {category.upper()} ({int(dist_now)}m)"
                    else:
                        box_color = (0, 165, 255)
                        label = f"WARNING ID:{tid} {category.upper()} ({int(dist_now)}m)"
                    if plate_tag:
                        label += f" [{plate_tag}]"
                else:
                    box_color = (0, 255, 0)
                    label = f"ID:{tid} {category}"

                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    box_color,
                    2,
                )
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    label,
                    (int(box[0]), max(15, int(box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    2,
                )

        cv2.putText(
            frame,
            "LIVE BRTS SURVEILLANCE FEED",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Render the main PMPML BRTS monitoring dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
def video_feed():
    """Stream MJPEG frames for the dashboard video view."""
    return StreamingResponse(
        generate_live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Save an uploaded video and switch the live feed to that file."""
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reset_session_state(str(file_path))
    return JSONResponse({"status": "success", "filename": file.filename})


@app.post("/api/use_webcam")
def use_webcam():
    """Switch the live feed to the local webcam device."""
    reset_session_state(0)
    return JSONResponse({"status": "success", "source": "webcam"})


@app.post("/api/test_sms")
def trigger_test_sms(req: TestSMSRequest):
    """Trigger an on-demand entry warning (Stage 0) or fine SMS (Stages 1-3) for demonstrations."""
    stage_presets: Dict[int, Tuple[float, int, int]] = {
        0: (15.0, 0, 0),
        1: (penalty_policy.stage1_distance_m, penalty_policy.base_fine_stage1, 0),
        2: (penalty_policy.stage2_distance_m, penalty_policy.base_fine_stage2, 0),
        3: (
            penalty_policy.stage3_distance_m,
            penalty_policy.base_fine_stage3,
            penalty_policy.stage3_base_interest_inr,
        ),
    }
    dist_m, fine_inr, interest_inr = stage_presets.get(req.stage, stage_presets[1])
    owner_phone = resolve_owner_phone(req.plate_number)
    msg_text = sms_service.build_message(
        plate_number=req.plate_number,
        stage=req.stage,
        distance_m=dist_m,
        fine_inr=fine_inr,
        interest_inr=interest_inr,
    )

    if sms_service.provider == "TWILIO" and sms_service.twilio_sid and sms_service.twilio_token and sms_service.twilio_from:
        from twilio.rest import Client  # type: ignore

        client = Client(sms_service.twilio_sid, sms_service.twilio_token)
        msg = client.messages.create(body=msg_text, from_=sms_service.twilio_from, to=owner_phone)
        logger.info("Demo SMS sent via Twilio SID=%s to %s", msg.sid, owner_phone)
        status = "SENT VIA TWILIO"
    else:
        logger.info(
            "[SMS DISPATCHED | DEMO] To: %s | Stage: %d | Plate: %s | Msg: %s",
            owner_phone,
            req.stage,
            req.plate_number,
            msg_text,
        )
        status = "DELIVERED (SIMULATED)"

    entry = record_sms_log(
        track_id=req.track_id,
        plate_number=req.plate_number,
        owner_phone=owner_phone,
        stage=req.stage,
        distance_m=dist_m,
        fine_inr=fine_inr,
        interest_inr=interest_inr,
        status=status,
    )
    return {"status": "sent", "sms": entry}


def _attach_evidence_url(row: dict) -> dict:
    """Attach evidence snapshot URL if a matching snapshot exists on disk."""
    eid = row.get("event_id")
    if eid and eid in evidence_by_event:
        row["evidence_url"] = evidence_by_event[eid]
        return row
    if eid:
        matches = sorted(EVIDENCE_DIR.glob(f"{eid}_stage*.jpg"), reverse=True)
        if matches:
            url = f"/evidence/{matches[0].name}"
            evidence_by_event[eid] = url
            row["evidence_url"] = url
    return row


@app.get("/api/violations")
def get_violations():
    """Return live violations (only >= 250m) and dispatched SMS logs."""
    db_violations = load_recent_violations_from_db(limit=10) if DB_AVAILABLE else []
    db_sms = load_recent_sms_from_db(limit=20) if DB_AVAILABLE else []
    total_sms_sent = load_total_sms_count_from_db() if DB_AVAILABLE else len(recent_sms_logs)

    if db_violations:
        enriched = [_attach_evidence_url(dict(r)) for r in db_violations]
        return {
            "total_violations": load_total_violations_count_from_db(),
            "total_sms_sent": total_sms_sent,
            "recent_violations": enriched,
            "sms_logs": db_sms if db_sms else recent_sms_logs[:20],
        }

    serialized_violations = [
        _attach_evidence_url(
            {
                "event_id": v.event_id,
                "track_id": v.track_id,
                "category": v.category,
                "plate_number": v.plate_number,
                "plate_confidence": v.plate_confidence,
                "timestamp": v.timestamp,
                "distance_m": v.distance_m,
                "stage": v.stage,
                "fine_amount": v.fine_amount,
                "interest_amount": v.interest_amount,
                "total_fine": v.total_fine,
                "status": v.status,
            }
        )
        for v in tracker_state.violations[:10]
    ]
    return {
        "total_violations": len(tracker_state.violations),
        "total_sms_sent": total_sms_sent,
        "recent_violations": serialized_violations,
        "sms_logs": db_sms if db_sms else recent_sms_logs[:20],
    }


@app.get("/api/challan/{event_id}/pdf")
def download_challan_pdf(event_id: str):
    """Generate and stream a downloadable PDF e-challan for a given violation event ID."""
    target = None

    for v in tracker_state.violations:
        if v.event_id == event_id:
            target = {
                "event_id": v.event_id,
                "track_id": v.track_id,
                "category": v.category,
                "plate_number": v.plate_number,
                "timestamp": v.timestamp,
                "distance_m": v.distance_m,
                "stage": v.stage,
                "fine_amount": v.fine_amount,
                "interest_amount": v.interest_amount,
                "total_fine": v.total_fine,
                "status": v.status,
                "evidence_path": v.evidence_path,
            }
            break

    if target is None and DB_AVAILABLE:
        for row in load_recent_violations_from_db(limit=100):
            if row.get("event_id") == event_id:
                target = dict(row)
                break

    if target is None:
        return JSONResponse({"error": "Violation event not found"}, status_code=404)

    if not target.get("evidence_path"):
        matches = sorted(EVIDENCE_DIR.glob(f"{event_id}_stage*.jpg"), reverse=True)
        if matches:
            target["evidence_path"] = str(matches[0])

    pdf_bytes = generate_challan_pdf_bytes(target)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="eChallan_{event_id}.pdf"'},
    )