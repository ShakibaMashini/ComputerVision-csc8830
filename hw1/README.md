# ComputerVision-csc8830-hw1
CSC 8830 – Computer Vision
Assignment 1: Measuring Real-World Object Dimensions Using Perspective Projection
# part 1
This repository contains my implementation for Assignment 1, which involves:
1. Computing real-world dimensions of an object from a single calibrated image using perspective projection equations.
2. Building a working application that allows a user to measure real-world distances by clicking two points on an object in the image.
3. Validating the measurements experimentally by imaging an object at a known distance and comparing the estimated size with the true measured size.

# Problem Description
Given a calibrated camera and an object placed at a known distance Zc from the camera, the goal is to:
* Click two image points corresponding to real points on the object
* Convert pixel coordinates into 3D rays using the intrinsic matrix K
* Back-project the rays to the known depth Zc
* Compute the Euclidean distance between the reconstructed 3D points
* Report this as the real-world distance (e.g., diameter, side length)

# Validation Experiment
To validate the measurement:
* Place an object at a known, accurately measured distance from the camera (e.g., 20 cm).
* Photograph the object.
* Run the script and click two physical points on the object.
* Compare the estimated vs. true distance.
# input Photo
![book](https://github.com/user-attachments/assets/c27ed9d8-b565-46a0-9d81-ad06c54710d5)

# Result Demo 
<img width="500" height="600" alt="measure_depth_result" src="https://github.com/user-attachments/assets/b4f4dba3-94e6-4cb3-b68b-418cdd6c9e8f" />

# part 2
# Web Application 
* index.html — browser interface
* script.js — click-based measurement logic
* style.css — UI styling

How to run the web app
* Simply open:
* web_app/index.html
* in any browser (Chrome, Safari, Firefox, Edge).
*  No installation is needed.
  
The web app:

* Loads an image
* Lets the user click two points
* Computes the real-world distance using the same projection equations
* Displays the output overlay on the image

<img width="1493" height="906" alt="Screenshot 2025-12-02 at 9 52 13 AM" src="https://github.com/user-attachments/assets/44b49c9d-5acb-408f-a6a8-e41822b0d5ca" />

This satisfies the requirement:
“Application must run as a Web application on a browser and be OS agnostic.”
