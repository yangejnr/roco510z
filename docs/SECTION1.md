# Section 1: Droplet Detection, Counting, and Blob COM

This repository implements **Section 1** of the coursework brief:

- Recognize **outer wrap** droplets from video
- Recognize **inner droplet** inside each outer wrap
- **Count successfully formed droplets**
- Perform **centre of mass / blob detection**

Implementation entry point: `droplet_section1.py`

## What The Script Produces

Given a video, the script writes:

- `summary.json`: aggregate metrics (including successful droplet count)
- `detections.csv`: per-frame detections (outer + inner geometry, blob COMs)
- `tracks.csv`: per-track summary (used for unique droplet counting)
- `annotated.mp4`: overlay video (optional) with circles + COM points

All coordinates are in **full-frame pixel coordinates** even if `--roi` / `--auto-roi` is used.

## Detection Pipeline (High Level)

Each frame is processed as follows:

1. Optional crop with `--roi` or motion-estimated `--auto-roi`.
1. Convert to grayscale and Gaussian blur for stability.
1. **Outer wrap detection**: `cv2.HoughCircles` finds circle candidates.
1. Rank candidates by edge energy on a thin ring around the circle and apply simple non-maximum suppression to remove near-duplicates.
1. For each outer circle, compute a binary mask inside the circle using Otsu thresholding (used for blob COM).
1. **Inner droplet detection**: search for a circle inside the outer wrap ROI (Hough, plus contour fallback). Inner max radius is capped to `0.9 * outer_r`.
1. **Blob centre-of-mass**: for outer and inner binary masks, find the largest connected component and compute its centroid using connected-components statistics.
1. Determine per-frame `success` using simple geometric checks (inner present, inner within outer, ratio bounds, near-concentric).

## Blob Detection / Centre Of Mass

“Blob detection” is implemented as:

- Create a binary image inside the outer/inner circular mask using Otsu thresholding.
- Run `cv2.connectedComponentsWithStats`.
- Select the **largest non-background component**.
- Use that component’s centroid as the **blob COM** and store its pixel area.

This is more explicit than raw moment-centroids over all foreground pixels, and gives you both:

- `com_*_x, com_*_y`: blob centroid
- `blob_*_area_px`: blob area in pixels

## Counting Successfully Formed Droplets

The script counts unique droplets using a simple tracker:

- Each detected outer wrap is assigned to a track by nearest-neighbor matching in image space (`--match-dist`).
- A track becomes a “successful droplet” if `success_frames >= --min-success-frames`.
- Output fields:
  - `summary.json`: `successful_tracks` and `success_formed_count`
  - `tracks.csv`: `is_success` for each track

Optional alternative counting mode:

- If `--count-line-y >= 0`, the script counts a successful droplet **once when the tracked outer wrap crosses the horizontal line**.

## CLI Options That Matter Most

Use `python droplet_section1.py --help` for the full list. Practically:

- `--auto-roi`: recommended for these high-speed videos; reduces false detections.
- `--progress-every`: prints progress every N frames while running.
- `--quiet`: suppresses stdout output (useful for batch runs).
- `--outer-min-radius`, `--outer-max-radius`: constrain wrap size in pixels.
- `--inner-min-radius`, `--inner-max-radius`: constrain inner size; inner is also capped to `0.9 * outer_r`.
- `--max-outer-per-frame`: cap candidates per frame to limit false positives and runtime.
- `--outer-acc-thresh`, `--inner-acc-thresh`: Hough circle sensitivity (lower = more circles, more false positives).
- `--match-dist`, `--max-gap-frames`: tracking stability.
- `--min-success-frames`: required consecutive-ish frames to confirm a droplet.
- `--ratio-min`, `--ratio-max`: success bounds for `inner_r/outer_r`.
- `--max-center-offset-frac`: success bound for centroid offset (fraction of outer radius).

## Outputs: Schema

### `detections.csv`

| Column | Meaning |
| --- | --- |
| `frame` | Frame index (0-based) |
| `track_id` | ID assigned by tracker |
| `success` | 1 if this detection meets success geometry checks |
| `score` | Soft confidence score (used for debugging) |
| `center_offset_px` | Pixel distance between inner and outer circle centres (blank if no inner) |
| `inner_outer_ratio` | `inner_r/outer_r` (blank if no inner) |
| `blob_outer_area_px` | Area of largest connected component inside the outer mask |
| `blob_inner_area_px` | Area of largest connected component inside the inner mask |
| `outer_cx, outer_cy, outer_r` | Outer wrap circle parameters |
| `inner_cx, inner_cy, inner_r` | Inner droplet circle parameters (blank if missing) |
| `com_outer_x, com_outer_y` | Blob COM for outer mask |
| `com_inner_x, com_inner_y` | Blob COM for inner mask (blank if missing) |

### `tracks.csv`

| Column | Meaning |
| --- | --- |
| `track_id` | Track ID |
| `total_frames` | Frames this track was matched |
| `success_frames` | Frames within the track marked as `success=1` |
| `is_success` | 1 if `success_frames >= --min-success-frames` |
| `last_frame` | Last frame index seen |
| `blob_outer_area_px` | Last observed outer blob area |
| `blob_inner_area_px` | Last observed inner blob area |
| `outer_*` / `inner_*` / `com_*` | Last observed geometry + COM values |

### `summary.json`

Includes:

- `roi`: ROI used (if `--roi` or `--auto-roi`), in `[x, y, w, h]`
- `successful_tracks`: unique droplet tracks considered successful
- `success_formed_count`: counting result (equals `successful_tracks` unless `--count-line-y` is used)

## Reproducing Results On The Included Videos

These were the settings used during development:

```bash
. .venv/bin/activate

python droplet_section1.py --video droplets_.mp4 --out outputs/latest_droplets \
  --annotate --auto-roi \
  --outer-min-radius 10 --outer-max-radius 120 \
  --inner-min-radius 3 --inner-max-radius 90 \
  --hough-min-dist 25 --outer-acc-thresh 28 --inner-acc-thresh 18 \
  --max-outer-per-frame 5 --match-dist 45 --max-gap-frames 4 \
  --min-success-frames 3 --progress-every 50

python droplet_section1.py --video droplets_1.mp4 --out outputs/latest_droplets1 \
  --annotate --auto-roi \
  --outer-min-radius 10 --outer-max-radius 120 \
  --inner-min-radius 3 --inner-max-radius 90 \
  --hough-min-dist 25 --outer-acc-thresh 28 --inner-acc-thresh 18 \
  --max-outer-per-frame 5 --match-dist 45 --max-gap-frames 4 \
  --min-success-frames 3 --progress-every 50
```

Open `annotated.mp4` to visually validate that:

- Green outer circle indicates `success=1` for that detection.
- White dot is outer blob COM.
- Yellow dot is inner blob COM (if inner detected).

## Common Tuning Advice

- Too many circles: increase `--outer-acc-thresh`, lower `--max-outer-per-frame`, and use `--auto-roi`.
- Missing circles: decrease `--outer-acc-thresh` slightly and widen radius bounds.
- Inner droplet wrong: tighten `--inner-max-radius` and adjust `--ratio-min/--ratio-max`.
- Counting unstable: increase `--min-success-frames` and reduce `--match-dist` if multiple droplets appear close together.

## Limitations (Explicit)

- Hough-based circle detection assumes roughly circular wraps/droplets; strongly non-circular shapes will degrade.
- Tracker is greedy nearest-neighbor; if objects cross/overlap heavily, IDs may swap.
- Blob COM depends on thresholding; if lighting changes drastically, consider adding a background subtractor or adaptive thresholding.
