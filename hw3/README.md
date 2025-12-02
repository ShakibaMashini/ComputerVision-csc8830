# ComputerVision-csc8830-hw3

# Homework 3: Gradient Operators, LoG, Edge Keypoints & Corner Keypoints
This assignment contains two main components:
  1. Image Derivative Computation
  2. Keypoint Detection (Edge-based + Corner-based)
All outputs are automatically generated and saved for visualization.


# Part 1 — Gradient Magnitude, Gradient Angle & Laplacian of Gaussian (LoG)
This section computes three different derivative-based representations for every input image:
  * Gradient Magnitude (Sobel)
  * Gradient Orientation (Sobel angle, visualized in degrees)
  * Laplacian of Gaussian (LoG) for second-derivative edge detection
These operations highlight edge strength, edge direction, and intensity curvature.

# Input Example
A sample input image from the dataset:

![Brain10](https://github.com/user-attachments/assets/d86d293d-e109-4440-bc6e-65f9416461a1)

# Output Demo
Below are the generated outputs for the above sample image:

# Gradient Angle

<img width="400" height="300" alt="Brain10_grad_ang" src="https://github.com/user-attachments/assets/073a23af-6b60-4335-bc72-9461cfb125df" />


# Gradient Magnitude
<img width="400" height="300" alt="Brain10_grad_mag" src="https://github.com/user-attachments/assets/898f5560-dc27-4f92-86ff-3eba31d8ae32" />

# Laplacian Of Gaussian(LoG)
<img width="400" height="300" alt="Brain10_log" src="https://github.com/user-attachments/assets/923b6fc0-7b9b-465e-84f4-00641e37e828" />

# Method Summary (Part 1)
1. Convert input to grayscale
2. Compute Sobel gx, gy
3. Compute:
* Magnitude: sqrt(gx² + gy²)
* Angle: atan2(gy, gx) → mapped to 0°–180° → normalized
4. Apply Gaussian smoothing
5. Apply Laplacian to produce LoG
6. Normalize all outputs to 0–255
7. Save each result to output_task1/

# Part 2 — Edge Keypoint Detection & Corner Keypoint Detection
This part implements two simple keypoint detectors:
  * EDGE keypoints using gradient magnitude + non-maximum suppression
  * CORNER keypoints using the Harris response + threshold + NMS
Both results are overlaid on the original image for visualization.

## Input Example


![4965713215249648563](https://github.com/user-attachments/assets/2f3a8b55-039c-4bd3-89ca-5d76244e5fa4)


## Output Demo

# EDGE Keypoints

<img width="400" height="300" alt="4965713215249648563_edge_kp" src="https://github.com/user-attachments/assets/6590d0b8-3607-4b0e-bfec-3a889d8ae650" />

# CORNER Keypoints

<img width="400" height="300" alt="4965713215249648563_corner_kp" src="https://github.com/user-attachments/assets/9db658c6-9d65-4a11-93b7-3721222f370c" />

##  Method Summary (Part 2)
# EDGE Keypoint Detector
* Apply Gaussian blur
* Compute Sobel gradients gx, gy
* Compute gradient magnitude
* Threshold strong responses
* Perform 3×3 non-maximum suppression
* Mark resulting positions as edge keypoints
# CORNER Keypoint Detector
* Heavy Gaussian smoothing
* Compute Harris corner response
* Threshold high-response regions
* Apply 3×3 NMS
* Keep strongest corners only


# Output Directories
part1/output_task1/
part2/output_part2/
Visualize using cv2.drawKeypoints()

