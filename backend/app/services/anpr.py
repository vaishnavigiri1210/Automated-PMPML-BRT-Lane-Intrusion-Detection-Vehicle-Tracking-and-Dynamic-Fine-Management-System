"""Real-time number plate reader (ANPR) using EasyOCR on vehicle crops."""
import logging
import re
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("brts.anpr")

# Matches Indian vehicle registration plates such as MH12AB1234 or MH14A1234
INDIAN_PLATE_REGEX = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}")


class PlateReader:
    """Extracts and normalizes vehicle registration numbers from detected vehicle frames."""

    def __init__(self, min_confidence: float = 0.35, min_ocr_interval: float = 0.6) -> None:
        self.min_confidence = min_confidence
        self.min_ocr_interval = min_ocr_interval
        self._reader = None
        self._init_attempted = False

    def _ensure_reader(self) -> bool:
        """Lazy-load EasyOCR reader so server startup stays fast."""
        if self._reader is not None:
            return True
        if self._init_attempted:
            return False

        self._init_attempted = True
        try:
            import easyocr  # type: ignore

            # gpu=False ensures reliable execution on CPU or mixed local environments
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized successfully.")
            return True
        except Exception as exc:
            logger.warning("EasyOCR unavailable (%s). Falling back to UNREADABLE status.", exc)
            return False

    @staticmethod
    def _clean_and_extract_plate(raw_text: str) -> Optional[str]:
        """Normalize OCR output and return a valid Indian plate string if matched."""
        alnum = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        match = INDIAN_PLATE_REGEX.search(alnum)
        if not match:
            return None

        plate = match.group(0)
        # Format cleanly with spaces where readable, e.g., MH12AB1234 -> MH 12 AB 1234
        prefix = plate[:4]
        rest = plate[4:]
        if len(rest) >= 5:
            letters = rest[:-4]
            digits = rest[-4:]
            return f"{prefix[:2]} {prefix[2:]} {letters} {digits}".strip()
        return plate

    def read_plate(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[Optional[str], float]:
        """Crop lower vehicle region, run EasyOCR, and return (plate_text, confidence)."""
        if not self._ensure_reader() or self._reader is None:
            return None, 0.0

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        # Focus on the lower half of the vehicle box where number plates are typically mounted
        box_h = y2 - y1
        crop_y1 = y1 + int(box_h * 0.45)
        crop = frame[crop_y1:y2, x1:x2]

        if crop.size == 0:
            return None, 0.0

        # Upscale small crops slightly to improve character recognition
        if crop.shape[1] < 160:
            scale = 160.0 / float(crop.shape[1])
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), max(24, int(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            detections = self._reader.readtext(crop)
        except Exception as exc:
            logger.debug("OCR inference skipped for crop: %s", exc)
            return None, 0.0

        best_plate: Optional[str] = None
        best_conf: float = 0.0

        for _, text, conf in detections:
            conf_val = float(conf)
            if conf_val < self.min_confidence:
                continue
            candidate = self._clean_and_extract_plate(str(text))
            if candidate and conf_val >= best_conf:
                best_plate = candidate
                best_conf = conf_val

        return best_plate, best_conf