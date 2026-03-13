# ROCO510/ROCO510Z Coursework 1 (Section 1)

Implements Section 1 droplet recognition from a video:

- Inner droplet detection
- Outer wrap detection
- Counting unique successfully formed droplets
- Centre of mass / blob centroid detection

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
. .venv/bin/activate
python droplet_section1.py --video /path/to/droplets.mp4 --out outputs
```

If the video has a large static background, crop to the printing region (still outputs full-frame coordinates):

```bash
python droplet_section1.py --video /path/to/droplets.mp4 --out outputs --roi 100,50,800,600
```

If you do not know the ROI, enable motion-based ROI estimation:

```bash
python droplet_section1.py --video /path/to/droplets.mp4 --out outputs --auto-roi
```

Outputs:

- `outputs/detections.csv` per-frame detections (inner/outer circles + COM + success)
- `outputs/tracks.csv` per-track summary and success count
- `outputs/annotated.mp4` overlay video (optional)
- `outputs/summary.json` aggregate counts

## Notes

The coursework brief doesn’t specify exact video characteristics in the PDF, so the
CLI exposes the most important parameters (radius ranges, thresholds, tracking).
