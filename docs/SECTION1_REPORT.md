# ROCO510/ROCO510Z Coursework 1: Section 1 Report

## Title

Computer Vision Pipeline for Droplet Recognition, Blob Centre-of-Mass Estimation, and Successful Droplet Counting in High-Speed Printing Video

## Abstract

This report presents a classical computer vision approach for analysing high-speed video from a droplet-based printing process. The proposed system detects the droplet **outer wrap** and, where present, the **inner droplet**, estimates blob-based centres of mass using connected-component statistics, and counts the number of **successfully formed** droplets by tracking detections over time. The method is implemented in Python using OpenCV and is designed to be reproducible via a command-line interface with exposed parameters controlling circle detection sensitivity, size constraints, region-of-interest selection, tracking association, and success criteria. Evaluation on two provided videos demonstrates stable detection and tracking in a narrow printing region identified via motion-based ROI estimation, producing successful droplet counts of 32 and 39 respectively under a transparent geometric definition of “success”.

## 1. Introduction

Droplet formation is a critical quality determinant in droplet-based manufacturing processes. In high-speed video, droplet structures often appear as an outer boundary (wrap) with a distinguishable inner core. A robust analysis pipeline must reliably detect these structures despite motion blur, specular highlights, background clutter, and small-scale variations in droplet appearance. Beyond instantaneous detection, droplet formation quality is inherently temporal: a droplet should be counted once as a physical object, rather than multiple times across frames. Consequently, a complete solution requires (i) per-frame recognition of relevant structures, (ii) blob localisation via centre-of-mass estimation, and (iii) temporal association to support unique counting.

This work implements a classical, interpretable pipeline consistent with the assessment requirements for Section 1: recognition of inner droplet and outer wrap, counting successfully formed droplets, and centre-of-mass/blob detection.

## 2. Data and Experimental Setup

Two example videos are included in the repository:

- `droplets_.mp4` (1280x720, 30 fps, 581 frames)
- `droplets_1.mp4` (1280x720, 30 fps, 675 frames)

The processing pipeline is implemented in `droplet_section1.py` and executed through a CLI. Outputs are written to a user-specified directory, including per-frame detections (`detections.csv`), per-track summaries (`tracks.csv`), a run summary (`summary.json`), and an optional annotated overlay video (`annotated.mp4`).

## 3. Methodology

### 3.1 Region of Interest (ROI) Selection

Droplet activity occupies a small subregion of the full frame. Processing the full image increases the number of spurious edges and circle hypotheses, degrading both accuracy and runtime. Two ROI mechanisms are supported:

1. Manual ROI: `--roi x,y,w,h`.
1. Motion-based ROI estimation: `--auto-roi`.

The automatic ROI estimates motion saliency from the temporal standard deviation of pixel intensity over an initial subset of frames. Pixels exceeding a high quantile are thresholded to define a bounding region, which is padded and used for subsequent detection. All reported coordinates are mapped back to full-frame coordinates to ensure consistent outputs.

### 3.2 Outer Wrap Detection

The outer wrap is modelled as a circle candidate detected via the gradient Hough transform (`cv2.HoughCircles`) applied to a blurred grayscale image. Because Hough methods can return multiple near-duplicate circles, candidates are filtered by:

- An edge-energy score computed on a thin ring around each circle (Canny edge response).
- A simple non-maximum suppression heuristic that removes candidates whose centres lie within a fraction of an already accepted circle’s radius.
- A maximum number of outer hypotheses per frame (`--max-outer-per-frame`) to control computational cost.

The outer circle is parameterised by centre `(c_x, c_y)` and radius `r_o`.

### 3.3 Inner Droplet Detection

Given an outer wrap candidate, the inner droplet is searched only within the corresponding wrap neighbourhood. The inner detection uses a smaller-radius Hough search within the wrap ROI, with a fallback to contour-based detection following Otsu thresholding and morphological opening. To ensure physical plausibility, the inner radius is constrained relative to the outer radius:

- `r_i <= 0.9 * r_o`.

The inner droplet circle is parameterised by centre `(c_x^i, c_y^i)` and radius `r_i` when detected.

### 3.4 Blob Detection and Centre of Mass

For both outer and inner structures, blob-based centre-of-mass estimation is computed from a binarised image restricted to the corresponding circle mask. The method is:

1. Mask the grayscale frame with a filled circle.
1. Apply Otsu thresholding inside the mask to obtain a binary image.
1. Apply morphological opening to remove isolated noise.
1. Perform connected-components labelling with statistics (`cv2.connectedComponentsWithStats`).
1. Select the largest non-background component and compute its centroid.

The centroid is treated as the blob COM:

- `COM = (x_bar, y_bar)`.

Additionally, the component area `A_blob` is recorded in pixels, enabling downstream quality analysis (e.g., droplet size stability).

### 3.5 Success Criteria and Droplet Counting

Per-frame success is defined by transparent geometric plausibility checks. A detection is marked `success=1` when:

1. Inner droplet is present and smaller than outer wrap: `r_i < r_o`.
1. Inner-to-outer radius ratio lies in a specified interval:
   - `rho = r_i / r_o`, with `--ratio-min <= rho <= --ratio-max`.
1. Inner and outer centres are sufficiently aligned:
   - `d = ||(c_x^i, c_y^i) - (c_x, c_y)||_2`,
   - `d <= alpha * r_o` where `alpha = --max-center-offset-frac`.

For unique counting, detections are associated over time using a nearest-neighbour tracker with maximum association distance `--match-dist` and a limited gap tolerance `--max-gap-frames`. A track is counted as a successfully formed droplet if it contains at least `--min-success-frames` successful frames. This yields a single count per physical droplet rather than per-frame counts.

## 4. Results

The following command configuration was used to process the included videos (with overlay outputs enabled for qualitative verification):

```bash
. .venv/bin/activate

python droplet_section1.py --video droplets_.mp4 --out outputs/latest_droplets \
  --annotate --auto-roi \
  --outer-min-radius 10 --outer-max-radius 120 \
  --inner-min-radius 3 --inner-max-radius 90 \
  --hough-min-dist 25 --outer-acc-thresh 28 --inner-acc-thresh 18 \
  --max-outer-per-frame 5 --match-dist 45 --max-gap-frames 4 \
  --min-success-frames 3

python droplet_section1.py --video droplets_1.mp4 --out outputs/latest_droplets1 \
  --annotate --auto-roi \
  --outer-min-radius 10 --outer-max-radius 120 \
  --inner-min-radius 3 --inner-max-radius 90 \
  --hough-min-dist 25 --outer-acc-thresh 28 --inner-acc-thresh 18 \
  --max-outer-per-frame 5 --match-dist 45 --max-gap-frames 4 \
  --min-success-frames 3
```

The automatic ROI estimates for these videos (reported in `summary.json`) were:

- `droplets_.mp4`: ROI `[45, 298, 404, 113]`, successful count `32`.
- `droplets_1.mp4`: ROI `[50, 306, 1050, 111]`, successful count `39`.

Annotated videos provide qualitative evidence of detection correctness by overlaying the outer wrap circle, inner droplet circle (when present), and blob COM points. Per-frame and per-track numeric outputs enable inspection of radius ratios, centre offsets, and blob areas.

## 5. Discussion

The system satisfies the Section 1 requirements with an interpretable design. Circle-based modelling is appropriate where wraps appear approximately circular, and the ROI mechanism substantially reduces background-induced false positives. Blob-based COM estimation via largest connected component yields stable centroids and provides a meaningful area measure, which is useful for process monitoring.

However, several limitations are noted:

1. The Hough transform assumes strong circular edges; non-circular or partially occluded wraps may be missed or mislocalised.
1. Threshold-based blob segmentation (Otsu) may be sensitive to illumination changes; adaptive thresholding or background subtraction may improve robustness.
1. The greedy nearest-neighbour tracker may produce ID swaps in dense scenes or when objects cross; a probabilistic motion model (e.g., Kalman filtering) and global assignment (Hungarian algorithm) could improve association.
1. The definition of “success” is geometric and generic. If the printing process defines success more specifically (e.g., minimum wrap thickness, stability after a nozzle plane, or temporal persistence), the criteria should be revised accordingly.

## 6. Conclusion

This report described a reproducible computer vision pipeline for outer wrap detection, inner droplet detection, blob centre-of-mass estimation, and successful droplet counting in high-speed printing video. The approach combines Hough-based circle hypotheses with blob segmentation for COM extraction and a simple tracker for unique counting. Evaluation on two provided videos yields stable droplet counts and provides structured outputs to support further analysis and reporting.

## References

1. Duda, R. O. and Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *Communications of the ACM*, 15(1), 11-15.
1. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
1. Gonzalez, R. C. and Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

