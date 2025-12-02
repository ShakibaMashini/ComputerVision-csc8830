# ComputerVision-csc8830-hw1


This homework includes two main tasks:

(1) Python-based measurement script that computes the real-world distance between two points on an object using perspective projection equations and known camera intrinsics.

(2) A browser-based web application that replicates the same measurement functionality using image upload and mouse clicks.

Both implementations allow the user to click two points on an object, specify the camera–object distance, and estimate the real-world length.

🎯 Objectives

1. Python Script for Real-World Measurement
<img width="622" height="615" alt="measure_depth_result" src="https://github.com/user-attachments/assets/423756aa-5941-4c5d-9289-81cec110e9ff" />

Loads an image containing a known object or calibration target.
Rescales camera intrinsic matrix to match the image size.
Allows user to select two pixel points.
Back-projects rays into 3D using known depth ( Z_c ).
Computes Euclidean distance between the two 3D points.
Visualizes and saves the measurement result.


2. Web Application
<img width="1493" height="906" alt="Screenshot 2025-12-02 at 9 52 13 AM" src="https://github.com/user-attachments/assets/14e737e6-e8fe-4975-b993-f4d22c51b643" />

Runs in any modern browser and any operating system.
Allows image upload.
User clicks two points to define a measurement segment.
Displays the estimated real-world length based on the same geometric principles.
