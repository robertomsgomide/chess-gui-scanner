import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

#########################################
# AutoDetector.py
#########################################
class AutoDetector:
    """
    Automatically locate an 8×8 chessboard inside a screenshot using a cascade
    of computer vision techniques.
    """
    _PATTERN_SIZE: Tuple[int, int] = (7, 7)

    def detect_board(self, pil_img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Return ``(left, top, right, bottom)`` of the detected board or *None*.
        It first tries to find inner corners, and failing that, falls back to
        finding the largest square-like contour.
        """
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # --- Strategy 1: Find Inner Corners (Fast but requires clear view) ---
        corners_bbox = self._detect_via_corners(img_bgr)
        if corners_bbox:
            return corners_bbox

        # --- Strategy 2: Find Contours (More robust when pieces are present) ---
        contour_bbox = self._detect_via_contours(img_bgr)
        if contour_bbox:
            return contour_bbox

        return None

    def _detect_via_corners(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Internal method using cv2.findChessboardCorners."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self._PATTERN_SIZE, flags)

        if not found or corners is None:
            return None

        corners = corners.reshape(-1, 2)
        if len(corners) != self._PATTERN_SIZE[0] * self._PATTERN_SIZE[1]:
            return None

        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)

        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x == 0 or span_y == 0: return None
        square_size = int(round(max(span_x, span_y) / 6.0))

        left = int(max(min_x - square_size, 0))
        top = int(max(min_y - square_size, 0))
        right = int(min(max_x + square_size, img_bgr.shape[1]))
        bottom = int(min(max_y + square_size, img_bgr.shape[0]))
        
        return self._normalize_bbox(left, top, right, bottom)

    def _detect_via_contours(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Internal method using edge detection and contour analysis."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise, then find edges with more sensitive thresholds
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)

        # Dilate the edges to help close gaps in the board's border, making it a single object
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Find all contours in the edge map, not just the outermost ones
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        max_area = 0

        # Sort contours by area and check the top 20 largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for cnt in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(cnt, True)
            # Use a slightly smaller epsilon for a more precise polygon approximation
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # We are looking for a quadrilateral (a shape with 4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                if w <= 0 or h <= 0:
                    continue

                aspect_ratio = w / float(h)
                # Use the more accurate contourArea instead of w*h
                area = cv2.contourArea(cnt)

                # Filter candidates by aspect ratio and a sensible area range
                if (
                    0.9 < aspect_ratio < 1.1 and  # Must be square-ish
                    w > 150 and h > 150 and      # Must be at least 150px wide/high
                    area > 40000 and             # Must have an area of at least 40k pixels
                    area > max_area              # Must be the largest valid square found so far
                ):
                    max_area = area
                    best_candidate = (x, y, x + w, y + h)

        if best_candidate:
            return self._normalize_bbox(*best_candidate)

        return None

    def _normalize_bbox(self, left: int, top: int, right: int, bottom: int) -> Optional[Tuple[int, int, int, int]]:
        """Ensures the final bounding box has dimensions divisible by 8."""
        width = right - left
        height = bottom - top
        
        width -= width % 8
        height -= height % 8
        
        if width <= 0 or height <= 0:
            return None
            
        right = left + width
        bottom = top + height
        return left, top, right, bottom