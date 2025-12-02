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


<img width="960" height="1280" alt="4965713215249648563_edge_kp" src="https://github.com/user-attachments/assets/4b2b7158-d3b6-48e6-9f77-add39d5799b6" />

# CORNER Keypoints

<img width="960" height="1280" alt="4965713215249648563_corner_kp" src="https://github.com/user-attachments/assets/7fd718f1-c684-4466-8223-0b1b08ec39c1" />


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
* part1/output_task1/
* part2/output_part2/
* Visualize using cv2.drawKeypoints()
## Part 3 – Object Boundary Extraction (No ML / DL Allowed)
A classical image-processing pipeline is used to extract the object boundary:

1. Convert to grayscale and blur (noise reduction)
2. Apply Otsu threshold to separate foreground/background
3. Perform morphological closing + opening to clean and smooth the mask
4. Find contours and choose the largest contour as the object
5. Create:
* A binary mask of the object
* An overlay image with the boundary drawn in red


#Input Example
![4965713215249648563](https://github.com/user-attachments/assets/531e3005-4519-44a9-ae48-72a559ea16ac)

## Output Demo

# Object Boundary Overlay 

<img width="200" height="600" alt="4965713215249648563_boundary" src="https://github.com/user-attachments/assets/f994fbcd-54da-493c-86e3-603b92ad6795" />

# Binary Mask
<img width="200" height="600" alt="4965713215249648563_mask" src="https://github.com/user-attachments/assets/b3cfca51-3454-44f6-9b0b-e32077d0e039" />

# Part 4 – Object Segmentation of a Non-Rectangular Object Using ArUco Markers

In this final part, we segment a non-rectangular object by placing ArUco markers along its physical boundary and capturing multiple images from various angles and distances.
The segmentation uses only classical computer vision (OpenCV), no machine learning or deep learning.

Method Overview

1. Convert image to grayscale and apply light smoothing
2. Detect ArUco markers using OpenCV (trying multiple dictionaries automatically)
3. Gather all detected marker corner points
4. Compute a convex hull over all marker points → this forms the object boundary
5. Generate:
  * An image with detected markers drawn
  * A binary object mask
  * An overlay showing the red boundary
  * A visualization of marker sizes (px), estimated from their corner geometry



## Input Example
![4965713215249648542](https://github.com/user-attachments/assets/60426c69-9d62-4b53-b3d4-e2b4de3e9e78)


## Output Demo

# Detected ArUco Markers
<img width="400" height="300" alt="4965713215249648542_markers" src="https://github.com/user-attachments/assets/4b6fa200-aac0-4e13-ae3a-fdb1b4309081" />

# Binary Mask of Segmented Object
<img width="400" height="300"alt="4965713215249648542_mask" src="https://github.com/user-attachments/assets/28373fc9-5be5-4368-91f4-ca0d169e0495" />

# Object Boundary Overlay (Red Hull)
<img width="400" height="300" alt="4965713215249648542_boundary" src="https://github.com/user-attachments/assets/467eba4f-7892-4103-b913-72af2e8f3930" />


## Part 5 – Comparison of ArUco-Based Segmentation vs. SAM2 Segmentation

In this final experiment, we compare the classical ArUco-marker-based segmentation from Part 4 with a modern foundation model, SAM2 (Segment Anything Model v2).
The purpose of this comparison is to evaluate:

* How accurately SAM2 can segment a non-rectangular object without markers
* How reliable ArUco-based geometric segmentation is when markers lie exactly on the boundary
* How the two masks differ in terms of boundary precision and coverage

# Comparison Approach

1. Use ArUco-based segmentation from Part 4 to produce:
  
  * Marker visualization
  * Binary mask
  * Red boundary overlay
  * Size estimation of markers
2. Run the SAM2 model on the same image to generate an automatic segmentation mask.

3. Create a comparison overlay to visualize differences:
  
  * White/green region → overlap
  * Red region → SAM2 over-segmentation
  * Blue region → SAM2 under-segmentation
  (depending on your visualization configuration)

# Input Example
** Note: (SAM2 is run on the same input used for ArUco segmentation)

# Output Demo

# Detected ArUco Markers

<img width="715" height="624" alt="4965713215249648543_markers" src="https://github.com/user-attachments/assets/bdf6d67f-612d-4816-a743-3cd7d27ce2e5" />

# ArUco Binary Mask

<img width="715" height="624" alt="4965713215249648543_mask_aruco" src="https://github.com/user-attachments/assets/b5c1c8ae-23e3-46e4-8d72-219296b6849e" />

# SAM2 Segmentation Mask
<img width="715" height="624" alt="4965713215249648543_mask_sam2" src="https://github.com/user-attachments/assets/7d16d079-471e-498b-bdc3-f9a93332d796" />

# ArUco Boundary Overlay (Red Hull)
<img width="715" height="624" alt="4965713215249648543_boundary_aruco" src="https://github.com/user-attachments/assets/1940228f-e723-45e0-8a42-51e65528f77f" />

# ArUco Boundary + Marker Size Visualization
<img width="715" height="624" alt="4965713215249648543_size_aruco" src="https://github.com/user-attachments/assets/7db41bfc-2c38-4b34-bf39-e65482e36b1f" />

# ArUco vs SAM2 Comparison Overlay
<img width="715" height="624" alt="4965713215249648543_compare_overlay" src="https://github.com/user-attachments/assets/dffebb01-46fa-4f00-8142-78acd8a13e88" />

