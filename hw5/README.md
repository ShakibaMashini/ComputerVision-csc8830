# ComputerVision-csc8830-hw5

# Homework 5 — Real-Time Object Tracking
This assignment demonstrates three different real-time object-tracking approaches using OpenCV.
All modes share the same camera input pipeline and rely on CSRT as the tracking backend.

The three implemented tracking methods are:
  1. Marker-Based Tracking (ArUco)
  2. Markerless Tracking (ROI + CSRT)
  3. SAM2-Based Tracking (using NPZ mask initialization)



# 1. Marker-Based Tracking (ArUco)
This mode tracks an object by detecting ArUco markers placed on its surface.
The system:
detects markers frame-by-frame,
estimates their corner positions,
and constructs a tight bounding box around the object.

📷 Demo – ArUco Marker Tracking

<img width="500" height="400" alt="Screenshot 2025-12-01 at 6 09 09 PM" src="https://github.com/user-attachments/assets/7c9c9ab0-36f4-4f79-a9e5-5f89ee3b9c58" />

# 2. Markerless Tracking (ROI + CSRT)
This mode does not require any marker.
The user manually selects an ROI in the first frame, and the CSRT tracker follows the object across subsequent frames.

📷 Demo – Markerless Tracking

<img width="500" height="400" alt="Screenshot 2025-12-01 at 6 10 05 PM" src="https://github.com/user-attachments/assets/27ca434d-ae33-4b1b-90f6-56f76b1e1eab" />


# 3. SAM2-Based Tracking (NPZ Initialization)
For this mode, a segmentation mask generated offline using SAM2 is loaded from an .npz file.
The mask is then used to initialize the bounding box automatically, which the CSRT tracker follows in real time.


📷 Demo – SAM2 Tracking

<img width="300" height="200" alt="Screenshot 2025-12-01 at 6 10 53 PM" src="https://github.com/user-attachments/assets/19eab790-bd83-4d0f-a7b3-343267e651fa" />


