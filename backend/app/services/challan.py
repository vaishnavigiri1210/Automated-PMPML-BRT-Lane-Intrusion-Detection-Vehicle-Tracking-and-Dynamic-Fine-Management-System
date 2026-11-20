"""PDF e-Challan generator with SHA-256 evidentiary digest for PMPML BRTS violations."""
import hashlib
import io
import json
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
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
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_challan_pdf_bytes(violation: Dict[str, Any]) -> bytes:
    """Render a downloadable A4 PDF e-Challan and return raw PDF bytes."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    sha256_hash = compute_evidence_sha256(violation)
    fine_inr = int(violation.get("fine_amount", 500))
    interest_inr = int(violation.get("interest_amount", 0))
    total_inr = int(violation.get("total_fine", fine_inr + interest_inr))

    # Draw official PMPML Transit Blue header banner
    pdf.setFillColor(colors.HexColor("#1E3A8A"))
    pdf.rect(0, height - 42 * mm, width, 42 * mm, fill=1, stroke=0)

    # Draw emerald accent strip below header
    pdf.setFillColor(colors.HexColor("#059669"))
    pdf.rect(0, height - 44 * mm, width, 2 * mm, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(20 * mm, height - 18 * mm, "PMPML BRTS - AUTOMATED E-CHALLAN")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        20 * mm,
        height - 26 * mm,
        "Pune Mahanagar Parivahan Mahamandal Ltd. | Dedicated BRT Corridor Enforcement",
    )
    pdf.drawString(20 * mm, height - 33 * mm, "Statutory Evidentiary Record (Bharatiya Sakshya Adhiniyam)")

    # Challan metadata block
    y = height - 58 * mm
    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(20 * mm, y, f"Challan ID: #{violation.get('event_id', 'N/A')}")
    pdf.drawRightString(width - 20 * mm, y, f"Timestamp: {violation.get('timestamp', 'N/A')}")

    y -= 10 * mm
    pdf.setStrokeColor(colors.HexColor("#D1D5DB"))
    pdf.line(20 * mm, y, width - 20 * mm, y)

    # Violation and vehicle details
    y -= 12 * mm
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(20 * mm, y, "Vehicle & Corridor Intrusion Details")

    rows = [
        ("Registration Plate", str(violation.get("plate_number", "UNREADABLE"))),
        ("Vehicle Category", str(violation.get("category", "UNKNOWN"))),
        ("Tracker Object ID", str(violation.get("track_id", 0))),
        ("Corridor Distance Traveled", f"{violation.get('distance_m', 0.0)} meters"),
        ("Escalation Stage", f"Stage {violation.get('stage', 1)}"),
        ("Enforcement Status", str(violation.get("status", "CONFIRMED_VIOLATION"))),
    ]

    pdf.setFont("Helvetica", 10)
    for label, val in rows:
        y -= 8 * mm
        pdf.setFillColor(colors.HexColor("#4B5563"))
        pdf.drawString(22 * mm, y, f"{label}:")
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(85 * mm, y, val)
        pdf.setFont("Helvetica", 10)

    # Penalty breakdown
    y -= 14 * mm
    pdf.setFillColor(colors.HexColor("#1E3A8A"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(20 * mm, y, "Penalty & Interest Summary")

    y -= 8 * mm
    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.setFont("Helvetica", 10)
    pdf.drawString(22 * mm, y, f"Base Fine: Rs. {fine_inr}")

    y -= 7 * mm
    pdf.drawString(22 * mm, y, f"Accrued Interest: Rs. {interest_inr}")

    y -= 9 * mm
    pdf.setFillColor(colors.HexColor("#DC2626"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(22 * mm, y, f"Total Payable Penalty: Rs. {total_inr}")

    # Cryptographic integrity digest block
    y -= 16 * mm
    pdf.setStrokeColor(colors.HexColor("#059669"))
    pdf.setFillColor(colors.HexColor("#ECFDF5"))
    pdf.rect(20 * mm, y - 14 * mm, width - 40 * mm, 20 * mm, fill=1, stroke=1)

    pdf.setFillColor(colors.HexColor("#065F46"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(24 * mm, y, "SHA-256 Evidentiary Integrity Digest:")
    pdf.setFont("Courier", 8)
    pdf.drawString(24 * mm, y - 7 * mm, sha256_hash)

    pdf.showPage()
    pdf.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes