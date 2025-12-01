# Homework 7 — Calibrated Stereo Size Estimation & Real-Time Pose/Hand Tracking

This assignment combines two main components:
1. Metric object-size estimation using a calibrated stereo camera pair, and
2. Real-time human-pose and hand-landmark tracking.
Both modules are accessible from my project’s main webpage and include demo images/videos as required.



# 1. Calibrated Stereo Object-Size Estimation
This part reimplements the earlier measurement pipeline from Assignment 1, but now uses full stereo calibration.
With known camera parameters, the depth of any matched point can be computed using:
Z = (f * B) / d
Where:
  * f — focal length
  * B — baseline distance between the cameras
  * d — disparity between corresponding pixels
After recovering the 3D point, the program measures real-world distances by computing pairwise 3D Euclidean lengths.
# Supported Object Types
  * Rectangular objects → width × height
  * Circular objects → diameter
  * General polygons → lengths of all visible edges
# Demo (Shown on Webpage)
  * Stereo feature matches
  * 3D reconstruction
  * Measured dimensions rendered directly on the image

# 2. Real-Time Pose Estimation & Hand Tracking
This module performs live body-pose and hand-landmark detection using either MediaPipe or OpenPose.
# Features Implemented
  * Real-time skeleton overlay
  * Real-time hand-landmark tracking
  * Visual output displayed on screen
  * Keypoint data saved frame-by-frame into a CSV file
  * No classification required — only keypoint extraction
# CSV Output Format
Each row corresponds to a single video frame and contains:
  * x, y coordinates for every detected joint/landmark
  * confidence scores
  * library-dependent missing-point indicators (0 or -1)
  This format allows further use in gesture analysis, biomechanics, or temporal modeling.
# Demo (Shown on Webpage)
  * Full skeleton visualization
  * Hand keypoints
  * Same data exported to CSV
