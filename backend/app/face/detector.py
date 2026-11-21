"""Face detection with OpenCV's Haar cascade (ships with opencv, no download needed).

Every image (for training and for verification) goes through the SAME function, so the
model always sees faces cropped, sized and coloured the same way.
"""
import io
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import settings

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MAX_SIDE = 1280          # big phone photos are shrunk first: faster and enough for detection
SMALL_IMAGE_SIDE = 200   # images this small with no detected face are treated as already-cropped


@dataclass
class FaceResult:
    face: Optional[np.ndarray]      # RGB uint8, IMG_SIZE x IMG_SIZE
    box: Optional[tuple]            # (x, y, w, h) in the resized image
    found: bool                     # True if a face was actually detected
    reason: str = ""


def load_rgb(image_bytes: bytes) -> np.ndarray:
    """Read any common image format, fix phone rotation (EXIF), return an RGB array."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Not a valid image file") from exc
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img)


def _detect_boxes(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(30, min(gray.shape[:2]) // 10)
    for neighbours in (5, 3):          # strict first, then a bit looser
        boxes = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=neighbours,
                                          minSize=(min_side, min_side))
        if len(boxes):
            return boxes
    return []


def crop_square(rgb: np.ndarray, box, margin: float = 0.2, size: int = None) -> np.ndarray:
    """Square crop around the box with some margin, resized to size x size."""
    size = size or settings.IMG_SIZE
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * margin))
    cx, cy = x + w // 2, y + h // 2
    H, W = rgb.shape[:2]
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(W, x1 + side), min(H, y1 + side)
    crop = rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def extract_face(image_bytes: bytes, allow_precropped: bool = False) -> FaceResult:
    """Find the largest face and return it as a 100x100 RGB crop.

    allow_precropped: for training uploads, a small image with no detected face is assumed to
    already be a cropped face (some datasets are stored that way) and is used as a whole.
    """
    rgb = load_rgb(image_bytes)
    boxes = _detect_boxes(rgb)
    if len(boxes):
        box = max(boxes, key=lambda b: b[2] * b[3])          # largest face = main person
        return FaceResult(crop_square(rgb, box), tuple(int(v) for v in box), True)
    if allow_precropped and max(rgb.shape[:2]) <= SMALL_IMAGE_SIDE:
        h, w = rgb.shape[:2]
        return FaceResult(crop_square(rgb, (0, 0, w, h), margin=0), None, False,
                          "no face detected; used whole small image as a pre-cropped face")
    return FaceResult(None, None, False, "no face detected")


def profile_photo(image_bytes: bytes) -> Optional[bytes]:
    """A 300x300 JPEG portrait (face with a wider margin) for the employee card."""
    rgb = load_rgb(image_bytes)
    boxes = _detect_boxes(rgb)
    if len(boxes):
        crop = crop_square(rgb, max(boxes, key=lambda b: b[2] * b[3]), margin=0.6, size=300)
    else:
        h, w = rgb.shape[:2]
        side = min(h, w)
        crop = cv2.resize(rgb[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2],
                          (300, 300))
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG", quality=90)
    return buf.getvalue()
