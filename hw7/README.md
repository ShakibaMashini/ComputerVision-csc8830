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
# Demo 
  * Stereo feature matches
  * 3D reconstruction
  * Measured dimensions rendered directly on the image

<img width="500" height="400" alt="Screenshot 2025-12-01 at 5 53 01 PM" src="https://github.com/user-attachments/assets/2b704d3e-0e4f-4936-927d-201bc870c825" />

    

# 2. Real-Time Pose Estimation & Hand Tracking
This module performs live body-pose and hand-landmark detection using either MediaPipe or OpenPose.
# Features Implemented
  * Real-time skeleton overlay
  * Real-time hand-landmark tracking
  * Visual output displayed on screen
  * Keypoint data saved frame-by-frame into a CSV file
  * No classification required — only keypoint extraction
# Saved Data Format (CSV)
  * Each frame written to the CSV contains:
  * the x/y pixel or normalized coordinates for each detected keypoint
  * the associated confidence score
  * missing detections encoded per-library (typically 0 or –1)
 This format allows further use in gesture analysis, biomechanics, or temporal modeling.

# Demo

<img width="500" height="400" alt="Screenshot 2025-12-01 at 5 58 33 PM" src="https://github.com/user-attachments/assets/ee7f53e6-fb53-4734-8a9c-5cb11ec3b5c0" />

  * Full skeleton visualization
  * Hand keypoints
  * Same data exported to CSV
<img width="500" height="400" alt="Screenshot 2025-12-01 at 5 58 59 PM" src="https://github.com/user-attachments/assets/60983e8e-fdbe-4103-806d-a396814a078c" />


