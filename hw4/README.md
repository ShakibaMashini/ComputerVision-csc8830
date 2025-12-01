# ComputerVision-csc8830-hw4
# Homework 4 — Panorama Construction via Image Stitching
This assignment implements a full panorama-stitching pipeline using feature-based homography estimation.
The project also includes a comparison against a smartphone-generated panorama, as well as a complete SIFT implementation from scratch followed by RANSAC-based alignment.

# Part 1 — Input Frames (Captured Manually)
A set of images was collected using a handheld camera with sufficient overlap between consecutive frames.
These serve as the inputs to the stitching pipeline.
Input Images
<h3 align="center">Input Images</h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8cc59e3-cbe0-444c-ba30-c9b93d39bd5d" width="200">
  <img src="https://github.com/user-attachments/assets/d75116fe-627b-4984-a869-1865ffa363cf" width="200">
  <img src="https://github.com/user-attachments/assets/ece3f2bb-c151-4956-8688-c1246538443f" width="200">
  <img src="https://github.com/user-attachments/assets/011dea48-d45b-494b-98f6-93388ae1f00a" width="200">
</p>
The overlapping content allows the algorithm to detect correspondences and compute valid homographies.

 
# Part 2 — Feature-Based Image Stitching (Homography Panorama)
The panorama is created using a classical homography-based workflow:
# Pipeline Steps
1. SIFT Feature Detection
2. Descriptor Extraction
3. Feature Matching
4. RANSAC Homography Estimation
5. Image Warping
6. Final Blending
# Stitched Panorama (Python Implementation

<img width="600" height="600" alt="output_panorama" src="https://github.com/user-attachments/assets/e35cfc82-e2a8-43ba-a649-0b14714b0d1c" />

The resulting panorama merges the input frames into a continuous wide-angle view.

#Phone Panorama Comparison
For reference, a panorama captured using a mobile device is shown below:

![phone_panorama](https://github.com/user-attachments/assets/816bee3f-944e-4a63-807a-5467a4d5266a)

Smartphone stitching generally includes:
* exposure compensation
* seam smoothing
* more aggressive blending
The Python version focuses on correctness and transparent feature-based alignment.


# Part 2 — SIFT From Scratch + RANSAC Alignment
This component implements the SIFT algorithm manually and compares its performance with OpenCV’s official version.

## Input Images
These two images were used for SIFT extraction and homography estimation:

<p align="center">
  <img src="https://github.com/user-attachments/assets/e444e50b-3547-4917-9aea-d8179f13cdf1" width="350">
  <img src="https://github.com/user-attachments/assets/8bb7d3a2-6d32-4bbc-9b0e-41a2dc5b5e5a" width="350">
</p>

# SIFT From Scratch — Full Pipeline
The implementation is based on the original SIFT paper (Lowe, 2004).
1. Gaussian Pyramid
* Construct multiple octaves
* Apply progressive blur using predetermined σ values
2. DoG Pyramid
* Subtract adjacent Gaussian levels
* Enhances blob-like structures
3. Keypoint Detection
* Locate extrema in a 3×3×3 neighborhood
* Apply contrast filtering
* Remove edge-like points via Hessian ratio test
4. Orientation Assignment
* Compute gradient magnitudes and orientations
* Build orientation histograms
* Assign canonical orientation for invariance
5. 128-D Descriptor Construction
* 4×4 spatial bins
* 8 orientation bins per cell
* Total: 128-dimensional descriptor
6. Feature Matching
* Euclidean distance between descriptors
* Lowe’s ratio test for outlier rejection
7. RANSAC Homography Estimation
* Robustly estimate transformation
* Remove mismatches

# Feature Matching — Results
 * OpenCV SIFT (Baseline)
<img width="300" height="200" alt="opencv_matches" src="https://github.com/user-attachments/assets/7be3b049-d134-48b3-80b6-af55e033642b" />

* Custom SIFT Implementation (From Scratch)
<img width="300" height="200" alt="ours_matches" src="https://github.com/user-attachments/assets/29d8a9cd-511a-4449-b44b-60df3e47517a" />

Comparison & Observations
# OpenCV SIFT:
  * Detects more interest points
  * Produces denser correspondences
# Custom SIFT Implementation:
  * Correctly identifies stable keypoints
  * Generates fewer but consistent matches
  * Performs well after RANSAC refinement
# Differences arise due to:
  * simplified scale-space
  * thresholding differences
  * numerical precision
  * gradient smoothing
Despite being implemented manually, the custom version produces meaningful matches and aligns well with the OpenCV baseline.


