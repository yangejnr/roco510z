# ROCO510Z — Computer Vision & Machine Learning Coursework

A University of Plymouth MSc Robotics coursework repository covering two distinct applied AI tasks: **droplet recognition/tracking from video** and **Fashion-MNIST image classification**.

**Status:** Academic coursework / implemented analysis pipeline

## What this repository demonstrates

This work shows two complementary forms of machine perception:

1. extracting structured geometric events from video using classical computer vision; and
2. training/evaluating an image classifier on Fashion-MNIST.

Rather than presenting the coursework as a single opaque script, the repository includes implementation notes, technical documentation, report-style write-ups and generated outputs.

## Section 1 — Droplet recognition from video

The video-analysis pipeline detects and tracks droplet formation events, including:

- inner droplet detection
- outer wrap detection
- centre-of-mass / blob centroid estimation
- unique droplet tracking
- successful formation counting
- optional region-of-interest cropping
- motion-assisted ROI estimation
- annotated video generation
- CSV/JSON result export

### Pipeline

```mermaid
flowchart LR
    VIDEO[High-speed Video] --> PRE[Pre-processing / ROI]
    PRE --> DETECT[Droplet & Wrap Detection]
    DETECT --> TRACK[Frame-to-frame Tracking]
    TRACK --> METRICS[Centroid / Formation Metrics]
    METRICS --> OUT[CSV + JSON + Annotated Video]
```

Run:

```bash
python droplet_section1.py --video /path/to/droplets.mp4 --out outputs
```

With automatic ROI estimation:

```bash
python droplet_section1.py --video /path/to/droplets.mp4 --out outputs --auto-roi
```

Outputs include:

- `outputs/detections.csv`
- `outputs/tracks.csv`
- `outputs/annotated.mp4`
- `outputs/summary.json`

## Section 2 — Fashion-MNIST classification

`fashion_mnist_section2.py` contains the machine-learning component of the coursework using the Fashion-MNIST dataset. The accompanying documentation explains the implementation and analysis separately from the droplet-vision task.

## Documentation

- `docs/SECTION1.md` — implementation notes
- `docs/SECTION1_REPORT.md` — report-style Section 1 write-up
- `docs/SECTION2.md` — Fashion-MNIST implementation
- `docs/SECTION2_TECHNICAL.md` — technical documentation
- `docs/SECTION2_REPORT.md` — report-style Section 2 write-up

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Engineering value

This repository is useful in the portfolio because it shows that my AI/robotics work is not limited to pretrained object detectors. The droplet task requires explicit image-processing, geometry, tracking and data extraction, while Fashion-MNIST demonstrates a separate supervised-learning workflow.

## Scope and limitations

This is academic coursework rather than a production application. Reported observations should be interpreted within the supplied datasets/videos and coursework methodology rather than as general production benchmarks.

## Author

**Yange Henry Terzugwe**  
Software Engineer | AI & Robotics  
MSc Robotics — University of Plymouth
