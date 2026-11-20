"""PDF e-Challan generator with embedded photographic evidence and SHA-256 digest."""
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def compute_evidence_sha256(violation: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash over violation details for integrity verification."""
    canonical = json.dumps(
        {
            "event_id": violation.get("event_id"),
            "track_id": violation.get("track_id"),
            "category": violation.get("category"),
            "plate_number": violation.get("plate_number"),
            "timestamp": violation.get("timestamp"),
            "distance_m": violation.get("distance_m"),
            "stage": violation.get("stage"),
            "fine_amount": violation.get("fine_amount"),
            "interest_amount": violation.get("interest_amount"),
            "total_fine": violation.get("total_fine"),
            "evidence_path": violation.get("evidence_path"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_evidence_file(evidence_path: Optional[str]) -> Optional[Path]:
    """Resolve an existing snapshot image file if available."""
    if not evidence_path:
        return None
    candidate = Path(evidence_path)
    if candidate.exists() and candidate.is_file():
        return candidate
    backend_relative = Path(__file__).resolve().parents[2] / evidence_path.lstrip("/\\")
    if backend_relative.exists() and backend_relative.is_file():
        return backend_relative
    return None


def generate_challan_pdf_bytes(violation: Dict[str, Any]) -> bytes:
    """Render a downloadable A4 PDF e-Challan with evidence snapshot and return raw PDF bytes."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    sha256_hash = compute_evidence_sha256(violation)
    fine_inr = int(violation.get("fine_amount", 500))
    interest_inr = int(violation.get("interest_amount", 0))
    total_inr = int(violation.get("total_fine", fine_inr + interest_inr))

    # Official PMPML Transit Blue header banner
    pdf.setFillColor(colors.HexColor("#0F2942"))
    pdf.rect(0, height - 40 * mm, width, 40 * mm, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#0D9488"))
    pdf.rect(0, height - 42 * mm, width, 2 * mm, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(18 * mm, height - 17 * mm, "PMPML BRTS - AUTOMATED E-CHALLAN")

    pdf.setFont("Helvetica", 9.5)
    pdf.drawString(
        18 * mm,
        height - 25 * mm,
        "Pune Mahanagar Parivahan Mahamandal Ltd. | Dedicated BRT Corridor Enforcement",
    )
    pdf.drawString(18 * mm, height - 32 * mm, "Statutory Evidentiary Record (Bharatiya Sakshya Adhiniyam)")

    # Challan ID & Timestamp row
    y = height - 54 * mm
    pdf.setFillColor(colors.HexColor("#0F172A"))
    pdf.setFont("Helvetica-Bold", 11.5)
    pdf.drawString(18 * mm, y, f"Challan ID: #{violation.get('event_id', 'N/A')}")
    pdf.drawRightString(width - 18 * mm, y, f"Timestamp: {violation.get('timestamp', 'N/A')}")

    y -= 7 * mm
    pdf.setStrokeColor(colors.HexColor("#CBD5E1"))
    pdf.line(18 * mm, y, width - 18 * mm, y)

    # Left column: Violation & Penalty details | Right column: Photographic Evidence
    top_section_y = y - 10 * mm
    left_x = 18 * mm
    right_box_x = 114 * mm
    right_box_w = width - right_box_x - 18 * mm
    right_box_h = 58 * mm

    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.setFillColor(colors.HexColor("#0F2942"))
    pdf.drawString(left_x, top_section_y, "Vehicle & Intrusion Summary")

    rows = [
        ("Plate Number", str(violation.get("plate_number", "UNREADABLE"))),
        ("Vehicle Type", str(violation.get("category", "UNKNOWN"))),
        ("Track ID", str(violation.get("track_id", 0))),
        ("Distance", f"{violation.get('distance_m', 0.0)} m"),
        ("Stage", f"Stage {violation.get('stage', 1)}"),
        ("Status", str(violation.get("status", "CONFIRMED_VIOLATION"))),
    ]

    curr_y = top_section_y - 7 * mm
    for label, val in rows:
        pdf.setFont("Helvetica", 9.5)
        pdf.setFillColor(colors.HexColor("#475569"))
        pdf.drawString(left_x, curr_y, f"{label}:")
        pdf.setFont("Helvetica-Bold", 9.5)
        pdf.setFillColor(colors.HexColor("#0F172A"))
        pdf.drawString(left_x + 31 * mm, curr_y, val)
        curr_y -= 6.5 * mm

    # Penalty block on left
    curr_y -= 5 * mm
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.setFillColor(colors.HexColor("#0F2942"))
    pdf.drawString(left_x, curr_y, "Penalty Assessment")

    curr_y -= 7 * mm
    pdf.setFont("Helvetica", 9.5)
    pdf.setFillColor(colors.HexColor("#0F172A"))
    pdf.drawString(left_x, curr_y, f"Base Fine: Rs. {fine_inr}")

    curr_y -= 6 * mm
    pdf.drawString(left_x, curr_y, f"Accrued Interest: Rs. {interest_inr}")

    curr_y -= 7.5 * mm
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.HexColor("#DC2626"))
    pdf.drawString(left_x, curr_y, f"Total Payable: Rs. {total_inr}")

    # Evidence Snapshot box on right
    ev_top_y = top_section_y + 2 * mm
    ev_bottom_y = ev_top_y - right_box_h
    pdf.setStrokeColor(colors.HexColor("#94A3B8"))
    pdf.setFillColor(colors.HexColor("#F8FAFC"))
    pdf.rect(right_box_x, ev_bottom_y, right_box_w, right_box_h, fill=1, stroke=1)

    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.setFillColor(colors.HexColor("#334155"))
    pdf.drawString(right_box_x + 3 * mm, ev_top_y - 5 * mm, "CORRIDOR EVIDENCE SNAPSHOT")

    ev_file = _resolve_evidence_file(violation.get("evidence_path"))
    if ev_file is not None:
        try:
            img_reader = ImageReader(str(ev_file))
            pdf.drawImage(
                img_reader,
                right_box_x + 3 * mm,
                ev_bottom_y + 3 * mm,
                width=right_box_w - 6 * mm,
                height=right_box_h - 10 * mm,
                preserveAspectRatio=True,
                anchor="c",
            )
        except Exception:
            pdf.setFont("Helvetica", 8)
            pdf.setFillColor(colors.HexColor("#64748B"))
            pdf.drawString(right_box_x + 4 * mm, ev_bottom_y + 24 * mm, "Snapshot render unavailable")
    else:
        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(colors.HexColor("#64748B"))
        pdf.drawString(right_box_x + 4 * mm, ev_bottom_y + 24 * mm, "No snapshot captured for event")

    # SHA-256 Evidentiary Digest box
    hash_y = min(curr_y - 16 * mm, ev_bottom_y - 16 * mm)
    pdf.setStrokeColor(colors.HexColor("#0D9488"))
    pdf.setFillColor(colors.HexColor("#F0FDFA"))
    pdf.rect(18 * mm, hash_y - 12 * mm, width - 36 * mm, 18 * mm, fill=1, stroke=1)

    pdf.setFillColor(colors.HexColor("#0F766E"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(22 * mm, hash_y + 1 * mm, "SHA-256 Evidentiary Integrity Digest:")
    pdf.setFont("Courier", 7.8)
    pdf.drawString(22 * mm, hash_y - 6.5 * mm, sha256_hash)

    pdf.showPage()
    pdf.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes