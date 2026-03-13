#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Detection:
    frame_idx: int
    outer: Circle
    inner: Optional[Circle]
    com_outer: tuple[float, float]
    com_inner: Optional[tuple[float, float]]
    blob_outer_area_px: int
    blob_inner_area_px: Optional[int]
    success: bool
    score: float


@dataclass
class Track:
    track_id: int
    last_frame: int
    last_outer: Circle
    total_frames: int = 0
    success_frames: int = 0
    counted: bool = False

    # Last known values for reporting
    last_inner: Optional[Circle] = None
    last_com_outer: tuple[float, float] = (math.nan, math.nan)
    last_com_inner: Optional[tuple[float, float]] = None
    last_blob_outer_area_px: int = 0
    last_blob_inner_area_px: Optional[int] = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def _moments_centroid(binary_mask: np.ndarray) -> Optional[tuple[float, float]]:
    m = cv2.moments(binary_mask, binaryImage=True)
    if m["m00"] <= 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _largest_component_com(binary_u8: np.ndarray) -> tuple[Optional[tuple[float, float]], int]:
    """
    Blob detection via connected components.

    Returns (centroid_xy, area_px) for the largest non-background component.
    """
    if binary_u8.dtype != np.uint8:
        binary_u8 = binary_u8.astype(np.uint8)
    _, bw = cv2.threshold(binary_u8, 0, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return (None, 0)

    # stats rows: [label, x, y, w, h, area] with row0 as background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    area = int(stats[idx, cv2.CC_STAT_AREA])
    cx, cy = centroids[idx]
    return ((float(cx), float(cy)), area)


def _circle_mask(shape_hw: tuple[int, int], circle: Circle) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(circle.cx)), int(round(circle.cy))), int(round(circle.r)), 255, -1)
    return mask


def _detect_circles_hough(
    gray_u8: np.ndarray,
    *,
    dp: float,
    min_dist: float,
    canny_hi: float,
    acc_thresh: float,
    min_r: int,
    max_r: int,
) -> list[Circle]:
    circles = cv2.HoughCircles(
        gray_u8,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=canny_hi,
        param2=acc_thresh,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    out: list[Circle] = []
    for (x, y, r) in circles[0]:
        out.append(Circle(float(x), float(y), float(r)))
    return out


def _rank_outer_circles(
    gray_u8: np.ndarray,
    circles: list[Circle],
    *,
    border_pad: int = 5,
    min_edge_score: float = 0.05,
    nms_center_dist_frac: float = 0.6,
) -> list[tuple[float, Circle]]:
    if not circles:
        return []

    h, w = gray_u8.shape[:2]
    edges = cv2.Canny(gray_u8, 80, 160)
    scored: list[tuple[float, Circle]] = []
    for c in circles:
        if (
            c.cx - c.r < border_pad
            or c.cy - c.r < border_pad
            or c.cx + c.r >= (w - border_pad)
            or c.cy + c.r >= (h - border_pad)
        ):
            continue
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (int(round(c.cx)), int(round(c.cy))), int(round(c.r)), 255, 2)
        score = float(np.mean(edges[ring > 0]) / 255.0)
        if score >= min_edge_score:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Simple NMS to avoid many near-duplicate circles around the same wrap.
    kept: list[tuple[float, Circle]] = []
    for score, c in scored:
        ok = True
        for _, kc in kept:
            dist = math.hypot(c.cx - kc.cx, c.cy - kc.cy)
            min_r = min(c.r, kc.r)
            if dist < (nms_center_dist_frac * min_r):
                ok = False
                break
        if ok:
            kept.append((score, c))
    return kept


def _detect_inner_in_outer(
    gray_u8: np.ndarray,
    outer: Circle,
    *,
    inner_min_r: int,
    inner_max_r: int,
    hough_dp: float,
    hough_min_dist: float,
    hough_canny_hi: float,
    hough_acc_thresh: float,
) -> Optional[Circle]:
    # Cap the inner search size based on the detected outer wrap.
    inner_max_r = min(int(inner_max_r), max(0, int(0.9 * outer.r)))
    if inner_max_r <= 0:
        return None

    h, w = gray_u8.shape[:2]
    x0 = _clamp_int(outer.cx - outer.r, 0, w - 1)
    y0 = _clamp_int(outer.cy - outer.r, 0, h - 1)
    x1 = _clamp_int(outer.cx + outer.r, 0, w - 1)
    y1 = _clamp_int(outer.cy + outer.r, 0, h - 1)
    roi = gray_u8[y0 : y1 + 1, x0 : x1 + 1]
    if roi.size == 0:
        return None

    roi_blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
    candidates = _detect_circles_hough(
        roi_blur,
        dp=hough_dp,
        min_dist=hough_min_dist,
        canny_hi=hough_canny_hi,
        acc_thresh=hough_acc_thresh,
        min_r=inner_min_r,
        max_r=inner_max_r,
    )
    if not candidates:
        # Fallback: threshold + contour + enclosing circle.
        roi2 = cv2.GaussianBlur(roi, (0, 0), 1.0)
        _, bw = cv2.threshold(roi2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[tuple[float, Circle]] = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < inner_min_r or r > inner_max_r:
                continue
            peri = cv2.arcLength(cnt, True)
            circ = 0.0 if peri <= 0 else float(4.0 * math.pi * area / (peri * peri))
            score = float(area) * circ
            if best is None or score > best[0]:
                best = (score, Circle(cx + x0, cy + y0, float(r)))
        return None if best is None else best[1]

    # Prefer the inner circle closest to the outer center (approx concentric).
    best = None
    for c in candidates:
        c_abs = Circle(c.cx + x0, c.cy + y0, c.r)
        d = math.hypot(c_abs.cx - outer.cx, c_abs.cy - outer.cy)
        key = d + 0.1 * c_abs.r
        if best is None or key < best[0]:
            best = (key, c_abs)
    return None if best is None else best[1]


def _success_criteria(
    outer: Circle,
    inner: Optional[Circle],
    *,
    max_center_offset_frac: float,
    ratio_min: float,
    ratio_max: float,
) -> tuple[bool, float]:
    if inner is None:
        return (False, 0.0)

    if inner.r <= 1.0 or outer.r <= 1.0:
        return (False, 0.0)

    if inner.r >= outer.r:
        return (False, 0.0)

    center_offset = math.hypot(inner.cx - outer.cx, inner.cy - outer.cy)
    offset_ok = center_offset <= (max_center_offset_frac * outer.r)
    ratio = inner.r / outer.r
    ratio_ok = float(ratio_min) <= ratio <= float(ratio_max)
    success = bool(offset_ok and ratio_ok)

    # Soft score for ranking/tracking confidence.
    score = float(max(0.0, 1.0 - center_offset / (max_center_offset_frac * outer.r + 1e-6)) * ratio)
    return (success, score)


def detect_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    *,
    outer_min_r: int,
    outer_max_r: int,
    inner_min_r: int,
    inner_max_r: int,
    hough_dp: float,
    hough_min_dist: float,
    hough_canny_hi: float,
    outer_acc_thresh: float,
    inner_acc_thresh: float,
    max_center_offset_frac: float,
    max_outer_per_frame: int,
    outer_fill_min: float,
    outer_fill_max: float,
    ratio_min: float,
    ratio_max: float,
) -> list[Detection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.5)

    outer_candidates = _detect_circles_hough(
        gray_blur,
        dp=hough_dp,
        min_dist=hough_min_dist,
        canny_hi=hough_canny_hi,
        acc_thresh=outer_acc_thresh,
        min_r=outer_min_r,
        max_r=outer_max_r,
    )
    ranked = _rank_outer_circles(gray_blur, outer_candidates)
    if not ranked:
        return []

    if max_outer_per_frame > 0:
        ranked = ranked[: max_outer_per_frame]

    dets: list[Detection] = []
    for _, outer in ranked:
        # Quick validation using the binary blob fill ratio inside the outer circle.
        outer_mask = _circle_mask(gray.shape[:2], outer)
        outer_pixels = cv2.bitwise_and(gray, gray, mask=outer_mask)
        _, outer_bw = cv2.threshold(outer_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        outer_bw = cv2.morphologyEx(outer_bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        area_mask = int(np.count_nonzero(outer_mask))
        area_fg = int(np.count_nonzero(outer_bw))
        if area_mask > 0:
            fill = area_fg / area_mask
            if not (outer_fill_min <= fill <= outer_fill_max):
                continue

        inner = _detect_inner_in_outer(
            gray_blur,
            outer,
            inner_min_r=inner_min_r,
            inner_max_r=inner_max_r,
            hough_dp=hough_dp,
            hough_min_dist=max(5.0, 0.5 * hough_min_dist),
            hough_canny_hi=hough_canny_hi,
            hough_acc_thresh=inner_acc_thresh,
        )

        com_outer, blob_outer_area_px = _largest_component_com(outer_bw)
        if com_outer is None:
            com_outer = (outer.cx, outer.cy)
            blob_outer_area_px = 0

        com_inner: Optional[tuple[float, float]] = None
        blob_inner_area_px: Optional[int] = None
        if inner is not None:
            inner_mask = _circle_mask(gray.shape[:2], inner)
            inner_pixels = cv2.bitwise_and(gray, gray, mask=inner_mask)
            _, inner_bw = cv2.threshold(inner_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inner_bw = cv2.morphologyEx(inner_bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            com, area = _largest_component_com(inner_bw)
            if com is not None:
                com_inner = com
                blob_inner_area_px = area

        success, score = _success_criteria(
            outer,
            inner,
            max_center_offset_frac=max_center_offset_frac,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
        )
        dets.append(
            Detection(
                frame_idx=frame_idx,
                outer=outer,
                inner=inner,
                com_outer=(float(com_outer[0]), float(com_outer[1])),
                com_inner=None if com_inner is None else (float(com_inner[0]), float(com_inner[1])),
                blob_outer_area_px=int(blob_outer_area_px),
                blob_inner_area_px=None if blob_inner_area_px is None else int(blob_inner_area_px),
                success=success,
                score=float(score),
            )
        )
    return dets


class DropletTracker:
    def __init__(
        self,
        *,
        match_dist_px: float,
        max_gap_frames: int,
    ) -> None:
        self._match_dist_px = float(match_dist_px)
        self._max_gap_frames = int(max_gap_frames)
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, dets: list[Detection]) -> list[Optional[int]]:
        if not dets:
            return []

        frame_idx = dets[0].frame_idx
        live_tracks = [t for t in self._tracks.values() if (frame_idx - t.last_frame) <= self._max_gap_frames]

        # Greedy nearest-neighbour matching: OK for low object counts.
        det_indices = list(range(len(dets)))
        unmatched_tracks = live_tracks[:]
        pairs: list[tuple[float, Track, Detection]] = []
        for t in unmatched_tracks:
            for i in det_indices:
                d = dets[i]
                dist = math.hypot(d.outer.cx - t.last_outer.cx, d.outer.cy - t.last_outer.cy)
                if dist <= self._match_dist_px:
                    pairs.append((dist, t, i))
        pairs.sort(key=lambda x: x[0])

        used_track_ids: set[int] = set()
        used_det_indices: set[int] = set()
        assignments: list[Optional[int]] = [None] * len(dets)
        for _, t, i in pairs:
            if t.track_id in used_track_ids:
                continue
            if i in used_det_indices:
                continue
            used_track_ids.add(t.track_id)
            used_det_indices.add(i)
            self._apply_update(t, dets[i])
            assignments[i] = t.track_id

        # New tracks for unmatched detections.
        for i in det_indices:
            if i in used_det_indices:
                continue
            tid = self._next_id
            self._next_id += 1
            d = dets[i]
            t = Track(track_id=tid, last_frame=d.frame_idx, last_outer=d.outer)
            self._tracks[tid] = t
            self._apply_update(t, d)
            assignments[i] = tid

        return assignments

    def _apply_update(self, track: Track, det: Detection) -> None:
        track.last_frame = det.frame_idx
        track.last_outer = det.outer
        track.last_inner = det.inner
        track.last_com_outer = det.com_outer
        track.last_com_inner = det.com_inner
        track.last_blob_outer_area_px = det.blob_outer_area_px
        track.last_blob_inner_area_px = det.blob_inner_area_px
        track.total_frames += 1
        if det.success:
            track.success_frames += 1


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {path}")
    return cap


def _writer_for(
    out_path: Path,
    *,
    fps: float,
    frame_size_wh: tuple[int, int],
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, frame_size_wh)


def _draw_detection(frame: np.ndarray, det: Detection, track_id: Optional[int]) -> None:
    outer = det.outer
    color_outer = (0, 200, 0) if det.success else (0, 0, 255)
    cv2.circle(frame, (int(round(outer.cx)), int(round(outer.cy))), int(round(outer.r)), color_outer, 2)

    if det.inner is not None:
        inner = det.inner
        cv2.circle(frame, (int(round(inner.cx)), int(round(inner.cy))), int(round(inner.r)), (255, 200, 0), 2)

    ox, oy = det.com_outer
    cv2.circle(frame, (int(round(ox)), int(round(oy))), 3, (255, 255, 255), -1)
    if det.com_inner is not None:
        ix, iy = det.com_inner
        cv2.circle(frame, (int(round(ix)), int(round(iy))), 3, (0, 255, 255), -1)

    label = f"id={track_id}" if track_id is not None else "id=?"
    cv2.putText(
        frame,
        label,
        (int(round(outer.cx + outer.r + 5)), int(round(outer.cy))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_outer,
        1,
        cv2.LINE_AA,
    )


def _write_detections_csv(path: Path, det_rows: Iterable[dict]) -> None:
    fieldnames = [
        "frame",
        "track_id",
        "success",
        "score",
        "center_offset_px",
        "inner_outer_ratio",
        "blob_outer_area_px",
        "blob_inner_area_px",
        "outer_cx",
        "outer_cy",
        "outer_r",
        "inner_cx",
        "inner_cy",
        "inner_r",
        "com_outer_x",
        "com_outer_y",
        "com_inner_x",
        "com_inner_y",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in det_rows:
            w.writerow(row)


def _write_tracks_csv(path: Path, tracks: list[Track], *, min_success_frames: int) -> None:
    fieldnames = [
        "track_id",
        "total_frames",
        "success_frames",
        "is_success",
        "last_frame",
        "blob_outer_area_px",
        "blob_inner_area_px",
        "outer_cx",
        "outer_cy",
        "outer_r",
        "inner_cx",
        "inner_cy",
        "inner_r",
        "com_outer_x",
        "com_outer_y",
        "com_inner_x",
        "com_inner_y",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tracks:
            inner = t.last_inner
            comi = t.last_com_inner
            w.writerow(
                {
                    "track_id": t.track_id,
                    "total_frames": t.total_frames,
                    "success_frames": t.success_frames,
                    "is_success": int(t.success_frames >= min_success_frames),
                    "last_frame": t.last_frame,
                    "blob_outer_area_px": int(t.last_blob_outer_area_px),
                    "blob_inner_area_px": "" if t.last_blob_inner_area_px is None else int(t.last_blob_inner_area_px),
                    "outer_cx": f"{t.last_outer.cx:.3f}",
                    "outer_cy": f"{t.last_outer.cy:.3f}",
                    "outer_r": f"{t.last_outer.r:.3f}",
                    "inner_cx": "" if inner is None else f"{inner.cx:.3f}",
                    "inner_cy": "" if inner is None else f"{inner.cy:.3f}",
                    "inner_r": "" if inner is None else f"{inner.r:.3f}",
                    "com_outer_x": f"{t.last_com_outer[0]:.3f}",
                    "com_outer_y": f"{t.last_com_outer[1]:.3f}",
                    "com_inner_x": "" if comi is None else f"{comi[0]:.3f}",
                    "com_inner_y": "" if comi is None else f"{comi[1]:.3f}",
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="ROCO510 Coursework 1 - Section 1 droplet detection")
    ap.add_argument("--video", required=True, help="Path to input video (e.g. .mp4)")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--annotate", action="store_true", help="Write annotated overlay video")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout status output")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N frames (0=off)")
    ap.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    ap.add_argument("--max-outer-per-frame", type=int, default=10, help="Max outer wrap candidates per frame (0=all)")
    ap.add_argument(
        "--roi",
        default="",
        help="Optional crop ROI as 'x,y,w,h' in pixels; detections are reported in full-frame coords",
    )
    ap.add_argument("--auto-roi", action="store_true", help="Estimate ROI from motion in the first frames")
    ap.add_argument("--auto-roi-frames", type=int, default=200)
    ap.add_argument("--auto-roi-step", type=int, default=5)
    ap.add_argument("--auto-roi-quantile", type=float, default=0.995)
    ap.add_argument("--auto-roi-pad", type=int, default=20)

    ap.add_argument("--outer-min-radius", type=int, default=20)
    ap.add_argument("--outer-max-radius", type=int, default=250)
    ap.add_argument("--inner-min-radius", type=int, default=5)
    ap.add_argument("--inner-max-radius", type=int, default=120)

    ap.add_argument("--hough-dp", type=float, default=1.2)
    ap.add_argument("--hough-min-dist", type=float, default=40.0)
    ap.add_argument("--hough-canny-hi", type=float, default=160.0)
    ap.add_argument("--outer-acc-thresh", type=float, default=25.0)
    ap.add_argument("--inner-acc-thresh", type=float, default=18.0)
    ap.add_argument("--outer-fill-min", type=float, default=0.0)
    ap.add_argument("--outer-fill-max", type=float, default=1.0)

    ap.add_argument("--max-center-offset-frac", type=float, default=0.25)
    ap.add_argument("--ratio-min", type=float, default=0.30, help="Success: inner_r/outer_r must be >= this")
    ap.add_argument("--ratio-max", type=float, default=0.90, help="Success: inner_r/outer_r must be <= this")
    ap.add_argument("--match-dist", type=float, default=60.0, help="Track matching distance in pixels")
    ap.add_argument("--max-gap-frames", type=int, default=2)
    ap.add_argument("--min-success-frames", type=int, default=3, help="Track must be successful for N frames")

    ap.add_argument(
        "--count-line-y",
        type=float,
        default=-1.0,
        help="Optional y pixel line; if set >=0 count a successful droplet when crossing it",
    )

    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    cap = _open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    writer = None
    if args.annotate:
        writer = _writer_for(out_dir / "annotated.mp4", fps=fps, frame_size_wh=frame_size)

    tracker = DropletTracker(match_dist_px=args.match_dist, max_gap_frames=args.max_gap_frames)
    det_rows: list[dict] = []

    count_line_y = float(args.count_line_y)
    formed_count = 0
    last_track_y: dict[int, float] = {}

    roi: Optional[tuple[int, int, int, int]] = None
    if args.roi:
        parts = [p.strip() for p in args.roi.split(",")]
        if len(parts) != 4:
            raise SystemExit("--roi must be 'x,y,w,h'")
        rx, ry, rw, rh = (int(float(p)) for p in parts)
        roi = (rx, ry, rw, rh)
    elif args.auto_roi:
        # Estimate ROI using temporal std-dev (motion) to narrow down detection.
        frames: list[np.ndarray] = []
        i = 0
        while i < max(1, int(args.auto_roi_frames)):
            ok, fr = cap.read()
            if not ok:
                break
            if int(args.auto_roi_step) <= 1 or (i % int(args.auto_roi_step) == 0):
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames.append(g)
            i += 1
        if frames:
            stack = np.stack(frames, axis=0)
            std = stack.std(axis=0)
            thr = float(np.quantile(std, float(args.auto_roi_quantile)))
            mask = (std >= thr).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if ys.size:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                pad = int(args.auto_roi_pad)
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(w - 1, x1 + pad)
                y1 = min(h - 1, y1 + pad)
                roi = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        # Reset to start for main processing.
        cap.release()
        cap = _open_video(args.video)

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
        if not args.quiet and int(args.progress_every) > 0 and (frame_idx % int(args.progress_every) == 0):
            print(f"[{Path(args.video).name}] frame {frame_idx}")

        frame_for_det = frame
        offx = 0
        offy = 0
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = _clamp_int(rx, 0, w - 1)
            ry = _clamp_int(ry, 0, h - 1)
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))
            frame_for_det = frame[ry : ry + rh, rx : rx + rw]
            offx, offy = rx, ry

        dets = detect_frame(
            frame_for_det,
            frame_idx,
            outer_min_r=args.outer_min_radius,
            outer_max_r=args.outer_max_radius,
            inner_min_r=args.inner_min_radius,
            inner_max_r=args.inner_max_radius,
            hough_dp=args.hough_dp,
            hough_min_dist=args.hough_min_dist,
            hough_canny_hi=args.hough_canny_hi,
            outer_acc_thresh=args.outer_acc_thresh,
            inner_acc_thresh=args.inner_acc_thresh,
            max_center_offset_frac=args.max_center_offset_frac,
            max_outer_per_frame=args.max_outer_per_frame,
            outer_fill_min=args.outer_fill_min,
            outer_fill_max=args.outer_fill_max,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
        )

        if offx or offy:
            adjusted: list[Detection] = []
            for d in dets:
                outer = Circle(d.outer.cx + offx, d.outer.cy + offy, d.outer.r)
                inner = None if d.inner is None else Circle(d.inner.cx + offx, d.inner.cy + offy, d.inner.r)
                com_outer = (d.com_outer[0] + offx, d.com_outer[1] + offy)
                com_inner = None if d.com_inner is None else (d.com_inner[0] + offx, d.com_inner[1] + offy)
                adjusted.append(
                    Detection(
                        frame_idx=d.frame_idx,
                        outer=outer,
                        inner=inner,
                        com_outer=com_outer,
                        com_inner=com_inner,
                        blob_outer_area_px=d.blob_outer_area_px,
                        blob_inner_area_px=d.blob_inner_area_px,
                        success=d.success,
                        score=d.score,
                    )
                )
            dets = adjusted

        assignments = tracker.update(dets)

        # Counting strategy: either count unique successful tracks, or (if a line is set)
        # count a successful track once when it crosses the line.
        for d, tid in zip(dets, assignments):
            if tid is None:
                continue
            cy = float(d.outer.cy)
            last_y = last_track_y.get(tid)
            last_track_y[tid] = cy
            if d.success and (tracker.tracks.get(tid) is not None):
                t = tracker.tracks[tid]
                is_success_track = t.success_frames >= args.min_success_frames
                if count_line_y >= 0 and last_y is not None and is_success_track and not t.counted:
                    if (last_y < count_line_y <= cy) or (last_y > count_line_y >= cy):
                        formed_count += 1
                        t.counted = True

        for d, tid in zip(dets, assignments):
            inner = d.inner
            comi = d.com_inner
            center_offset = "" if inner is None else f"{math.hypot(inner.cx - d.outer.cx, inner.cy - d.outer.cy):.3f}"
            ratio = "" if inner is None else f"{(inner.r / d.outer.r):.6f}"
            det_rows.append(
                {
                    "frame": d.frame_idx,
                    "track_id": "" if tid is None else tid,
                    "success": int(d.success),
                    "score": f"{d.score:.6f}",
                    "center_offset_px": center_offset,
                    "inner_outer_ratio": ratio,
                    "blob_outer_area_px": int(d.blob_outer_area_px),
                    "blob_inner_area_px": "" if d.blob_inner_area_px is None else int(d.blob_inner_area_px),
                    "outer_cx": f"{d.outer.cx:.3f}",
                    "outer_cy": f"{d.outer.cy:.3f}",
                    "outer_r": f"{d.outer.r:.3f}",
                    "inner_cx": "" if inner is None else f"{inner.cx:.3f}",
                    "inner_cy": "" if inner is None else f"{inner.cy:.3f}",
                    "inner_r": "" if inner is None else f"{inner.r:.3f}",
                    "com_outer_x": f"{d.com_outer[0]:.3f}",
                    "com_outer_y": f"{d.com_outer[1]:.3f}",
                    "com_inner_x": "" if comi is None else f"{comi[0]:.3f}",
                    "com_inner_y": "" if comi is None else f"{comi[1]:.3f}",
                }
            )

        if writer is not None:
            for d, tid in zip(dets, assignments):
                if tid is None:
                    continue
                _draw_detection(frame, d, tid)
            if count_line_y >= 0:
                cv2.line(frame, (0, int(round(count_line_y))), (w - 1, int(round(count_line_y))), (255, 255, 255), 1)
            if roi is not None:
                rx, ry, rw, rh = roi
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    # Aggregate success count (unique tracks meeting success frame threshold).
    tracks = list(tracker.tracks.values())
    success_tracks = [t for t in tracks if t.success_frames >= args.min_success_frames]
    success_count = len(success_tracks)
    if count_line_y < 0:
        formed_count = success_count

    _write_detections_csv(out_dir / "detections.csv", det_rows)
    _write_tracks_csv(out_dir / "tracks.csv", tracks, min_success_frames=args.min_success_frames)

    summary = {
        "video": os.path.abspath(args.video),
        "frames_processed": frame_idx + 1,
        "fps": float(fps),
        "frame_size_wh": [w, h],
        "roi": None if roi is None else [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
        "min_success_frames": int(args.min_success_frames),
        "successful_tracks": int(success_count),
        "success_formed_count": int(formed_count),
        "count_line_y": None if count_line_y < 0 else float(count_line_y),
        "outputs": {
            "detections_csv": str((out_dir / "detections.csv").resolve()),
            "tracks_csv": str((out_dir / "tracks.csv").resolve()),
            "annotated_mp4": None if not args.annotate else str((out_dir / "annotated.mp4").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(
            f"[{Path(args.video).name}] frames={summary['frames_processed']} "
            f"roi={summary['roi']} success_tracks={summary['successful_tracks']} "
            f"formed={summary['success_formed_count']} out={out_dir.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
