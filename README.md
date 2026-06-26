# Computer Vision — SNUCSE 2026

Course assignment repository for **Computer Vision** at Seoul National University (Spring 2026).

This course follows the arc of **"2D understanding → 3D reconstruction"**: starting from image formation and classical feature descriptors, progressing through recognition, tracking, and segmentation, and culminating in projective geometry, camera models, multi-view geometry, and 3D reconstruction.

All implementations are in **Python (Jupyter Notebook / `.py`)**.

And all works are partially helped by LLM(claude)^^

---

## Course Topics

**Midterm (2D world)** — What is in an image, and where?

| Lecture | Topic | Key Idea |
|---------|-------|----------|
| 2–3 | Linear Algebra Review | SVD as the universal solver for `Ax = 0` |
| 4–5 | Filtering & Edges | Convolution, Gaussian blur, Canny detector |
| 6–8 | Feature Descriptors | HOG (cell histograms), SIFT (DoG + 128-dim), Harris (eigenvalue corners) |
| 9 | Hough Transform | Voting in parameter space for lines/circles |
| 10–12 | Recognition | Bag of Words, face detection, PCA & Eigenface |
| 13–14 | Tracking & Segmentation | Lucas-Kanade optical flow, energy-minimization segmentation |

**Final (3D world)** — Reverse-engineer the scene that produced the image.

| Lecture | Topic | Key Idea |
|---------|-------|----------|
| 15–18 | Projective Geometry | Homogeneous coords, Homography `H` via DLT |
| 19–20 | Camera Model | `P = K[R\|t]`, pixel ↔ 3D ray |
| 21 | Calibration | Zhang's method (checkerboard → `K`), lens distortion |
| 22 | PnP | Recover `[R\|t]` from 3D-2D correspondences + RANSAC |
| 23 | Two-View Geometry | Fundamental matrix `F`, epipolar constraint, triangulation |
| 24 | SfM & Bundle Adjustment | Incremental reconstruction + joint reprojection-error minimization |
| 25–26 | Stereo & MVS | Disparity → depth, structured light, multi-view stereo |

---

## Assignment #1 — Classical Feature Descriptors

**Topics:** SVD, HOG, SIFT

### 1-1. SVD Image Compression (`HW1/1_svd/`) 🗜️

Decomposed a grayscale image using `numpy.linalg.svd` and reconstructed it using the top-k singular components. Implemented an interactive rank-k slider to visualize the compression-quality trade-off.

### 1-2. HOG-Based Face Detection (`HW1/2_hog/`) 🔍

Implemented HOG feature extraction from scratch: gradient computation (Sobel/Prewitt/Scharr), cell-wise orientation histograms, block normalization, and HOG visualization by overlaying gradient direction lines on the original image.

### 1-3. SIFT Feature Matching (`HW1/3_sift/`) 🔑

Built a SIFT pipeline from scratch: Gaussian pyramid construction, Difference-of-Gaussian (DoG) scale-space extrema detection, orientation assignment, and 128-dim descriptor computation. Compared with OpenCV's built-in SIFT for cross-image keypoint matching.

---

## Assignment #2 — Recognition Pipelines

**Topics:** Harris Corner Detection, Bag of Words / Scene Recognition, PCA & Eigenface

### 2-1. Harris Corner Detection (`HW2/1_harris/`) 📐

Implemented Harris corner detection: Sobel gradients → structure tensor → Harris response score → non-maximum suppression. Added an interactive visualization to inspect the local Harris matrix and eigenvalue decomposition at any clicked point.

### 2-2. Scene Recognition via Bag of Words (`HW2/2_scene_recognition/`) 🏙️

Extracted SIFT features from scene images → k-means clustering to build a visual vocabulary → represent each image as a BoW histogram → classify with **LinearSVC**. Implemented nearest-neighbor search for visual word assignment.

### 2-3. Eigenface (`HW2/3_eigenface/`) 👤

Centered the Olivetti face dataset and applied PCA via SVD. Projected faces into the eigenface space and reconstructed them from a reduced number of principal components. Compared different centering strategies and explored reconstruction quality vs. number of eigenvectors retained.

---

## Assignment #3 — Geometry Estimation

**Topics:** Hough Transform, Homography Estimation with RANSAC

### 3-1. Hough Transform (`HW3/hough/`) 📏

Implemented Hough line detection (ρ-θ accumulator) and Hough circle detection (3D accumulator over (a, b, r)) from scratch without using OpenCV's built-in functions. Applied non-maximum suppression on the accumulator for peak selection.

### 3-2. Homography Estimation & Image Stitching (`HW3/homography/`) 🧩

- **Q2-1:** Estimated homography from manually specified point correspondences (DLT, `Ax = 0` → SVD) and used it to insert an image into a target scene via projective warping.
- **Q2-2:** Automated the pipeline with SIFT feature extraction, ratio-test matching, and **RANSAC-based homography estimation**. Merged two overlapping photos into a panorama.

---

## Assignment #4 — 3D Human Pose Reconstruction

**Topics:** Multi-view Triangulation, Panoptic Studio-style 3D Pose Estimation

Based on the **CMU Panoptic Studio** framework. Given multi-view video and 2D heatmaps of human body keypoints, the task is to reconstruct 3D keypoint positions from calibrated cameras.

### 4-1. Two-View & N-View Triangulation (`Section 2`) 📐

Implemented DLT triangulation for a single person's keypoints from two calibrated camera views. Extended to N-view triangulation by stacking all available camera observations and solving the overdetermined system via SVD.

### 4-2. Multi-Person Reconstruction (`Section 3`) 👥

Extended single-person triangulation to the multi-person case. Handled the cross-view person-matching ambiguity via brute-force association before triangulating each person's skeleton independently.

### 4-3. Panoptic Studio Reconstruction Pipeline (`Section 4`) 🦴

Followed the full Panoptic Studio pipeline:
1. **Node (joint) proposals** from per-view 2D heatmaps via NMS
2. **Part (bone) proposals** from part-affinity fields
3. **Skeletal proposals** using dynamic programming + projection filter
4. **Joint refinement** to produce the final 3D skeleton estimates

A custom 3D viewer (`visualizer/panoptic_visualizer.py`) was used throughout for debugging and result inspection.

---

## Notes

- This repository is for **educational and academic purposes only**.
- HW1–HW3 notebooks run locally. HW4 was originally designed for Google Colab (GPU recommended for the panoptic section).
