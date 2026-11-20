# Automated PMPML - BRT Lane Intrusion Detection, Vehicle Tracking, and Dynamic Fine Management System

A real-time computer vision surveillance and automated e-challan enforcement engine designed for **Pune Mahanagar Parivahan Mahamandal Ltd. (PMPML) Bus Rapid Transit System (BRTS)** dedicated corridors.

---

## Core Capabilities

- **Real-Time Multi-Object Tracking:** Uses **YOLOv8** and **ByteTrack** to detect and track vehicles across live CCTV video streams, uploaded `.mp4` footage, or local webcams.
- **Corridor Polygon & Authorization Filtering:** Checks vehicle tire ground points against a dedicated BRT corridor polygon. Public transit buses and emergency vehicles are whitelisted, while unauthorized private vehicles are flagged.
- **Automated Number Plate Recognition (ANPR):** Uses throttled **EasyOCR** inference on detected violators to extract and normalize Indian registration plates (`MH12...`, `MH14...`).
- **4-Tier Progressive Enforcement & SMS Escalation:**
  - **Stage 0 (`0m` — Entry Warning):** Dispatches an immediate exit warning SMS (`₹0` fine) upon corridor entry and valid plate recognition.
  - **Stage 1 (`250m`):** Issues a **₹500** fine and Stage 1 SMS notice.
  - **Stage 2 (`1,000m`):** Escalates the fine to **₹1,200** with a Stage 2 SMS warning.
  - **Stage 3 (`1,500m+`):** Escalates the fine to **₹2,000 + accrued distance-based interest** with a critical e-challan SMS alert.
- **Persistent PostgreSQL Storage:** Stores all confirmed violations and SMS dispatch logs in PostgreSQL (`brts_db`) via SQLAlchemy with automatic in-memory fallback.
- **Tamper-Evident PDF E-Challans:** Generates downloadable A4 PDF e-challans (`ReportLab`) embedded with a **SHA-256 cryptographic integrity digest**.
- **Live Operations Dashboard:** Built with **FastAPI** and **Tailwind CSS**, featuring live MJPEG streaming, KPI counters, real-time violation cards, SMS logs, and on-demand stage test triggers.

---

## System Architecture

```text
Video Stream (CCTV / MP4 / Webcam)
        │
        ▼
YOLOv8 Detection + ByteTrack ID Persistence
        │
        ▼
BRT Lane Polygon Check & Authorization Whitelist (Bus / Ambulance vs. Private)
        │
        ▼
EasyOCR ANPR + Cumulative Corridor Distance Estimator
        │
        ├──► Stage 0 (0m): Immediate Exit Warning SMS (₹0)
        └──► Stage 1 (250m) ──► Stage 2 (1000m) ──► Stage 3 (1500m + Interest)
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
           PostgreSQL DB      Twilio / Sim SMS    SHA-256 PDF E-Challan
```

---

## Project Structure

```text
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI server, MJPEG stream & REST APIs
│       ├── services/
│       │   ├── anpr.py              # EasyOCR license plate reader
│       │   ├── authorization.py     # Public transit vs. private vehicle classifier
│       │   ├── challan.py           # ReportLab PDF generator with SHA-256 digest
│       │   ├── config.py            # Distance thresholds & fine policy config
│       │   ├── db.py                # PostgreSQL SQLAlchemy models & persistence
│       │   ├── intrusion.py         # Multi-frame intrusion & stage escalation tracker
│       │   ├── lane.py              # Ground-point polygon & distance estimation
│       │   └── sms.py               # Deduplicated stage-wise SMS alert service
│       └── templates/
│           └── index.html           # Live PMPML BRTS monitoring dashboard
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Clone & Setup Virtual Environment
```powershell
git clone [https://github.com/vaishnavigiri1210/Automated-PMPML---BRT-Lane-Intrusion-Detection-Vehicle-Tracking-and-Dynamic-Fine-Management-System.git](https://github.com/vaishnavigiri1210/Automated-PMPML---BRT-Lane-Intrusion-Detection-Vehicle-Tracking-and-Dynamic-Fine-Management-System.git)
cd Automated-PMPML---BRT-Lane-Intrusion-Detection-Vehicle-Tracking-and-Dynamic-Fine-Management-System

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure PostgreSQL Database
Create a PostgreSQL database named `brts_db`:
```sql
CREATE DATABASE brts_db;
```
*(Default connection string in `db.py`: `postgresql+psycopg2://postgres:1234@127.0.0.1:5432/brts_db`, configurable via `DATABASE_URL` environment variable.)*

### 3. Run the FastAPI Server
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access the dashboard.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, OpenCV
- **Computer Vision & OCR:** Ultralytics YOLOv8, ByteTrack, EasyOCR, PyTorch
- **Database:** PostgreSQL, SQLAlchemy, Psycopg2
- **Reporting & Alerts:** ReportLab (PDF), SHA-256 Hashing, Twilio / Simulated SMS Gateway
- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript