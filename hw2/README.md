# ComputerVision-csc8830-hw2
## Template Matching & Fourier-Based Image Restoration
This assignment consists of two main components:
  1. Object Detection using Template Matching (Correlation Method)
  2. Gaussian Blurring + Image Restoration using the Fourier Transform
All visual examples below are produced by the implemented Python pipeline.

# Part 1 — Object Detection Using Template Matching (Correlation)
This section implements object detection using correlation-based template matching.
Unlike crop-from-same-scene methods, each template is taken from a completely different image, which makes the task significantly more challenging.
✔ Requirements Addressed
* Templates not cropped from the test scene
* At least 10 different objects evaluated
* Objects may appear in the same or different scene images
* Correlation used as the matching metric
* Detected regions highlighted in output images

# Template Images (Examples)  
<p align="center">
  <img src="https://github.com/user-attachments/assets/d8876e5f-dc01-49fb-ade3-e385aa9bfd40" width="220">
  <img src="https://github.com/user-attachments/assets/461c1208-e045-456d-a307-4596dd178d20" width="220">
  <img src="https://github.com/user-attachments/assets/a6c816a2-efc6-473d-af5e-2b4ee0740ba9" width="220">
  <img src="https://github.com/user-attachments/assets/3ba30716-be63-4e91-a63b-b63861835160" width="220">
</p>

# Scene Images (Examples)
<p align="center">
  <img src="https://github.com/user-attachments/assets/a20bc036-3e7f-4e26-a084-ece11680f7bb" width="320">
  <img src="https://github.com/user-attachments/assets/b28dce40-e45f-4afa-8780-61ae8f10bd23" width="320">
</p>


# Detection Results(Examples)

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ea33e8c-0cde-42f7-a570-e521e709744c" width="280">
  <img src="https://github.com/user-attachments/assets/8d1c2b04-0831-4106-8694-e021de907d5f" width="280">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0c15882-cbd0-48cd-8e47-d527b01012e7" width="280">
  <img src="https://github.com/user-attachments/assets/1ee85409-b94c-4226-bb2a-95990cb9d541" width="280">
</p>

## Part 2 — Convolution & Fourier Transform (Blurring + Restoration)
In this part of the assignment, we explore Gaussian blurring and image restoration using the Fourier transform.

Steps required:

1. Capture an image ( L ) using any camera of choice.
2. Apply a Gaussian Blur to obtain the blurred image ( L_b ).
3. Attempt to recover the original image from ( L_b ) using frequency-domain filtering (Fourier Transform).
4. Demonstrate each step with visual results.

# Input Image (Original)
<img width="400" height="300" alt="Screenshot 2025-12-02 at 9 46 17 AM" src="https://github.com/user-attachments/assets/5ada61ca-8f3c-48e9-91d0-45cd98c19aa9" />


# Gaussian Blurred Image

<img width="400" height="300" alt="Nailpolish_blurred" src="https://github.com/user-attachments/assets/c900aa41-01e9-43ac-a33c-580feca49991" />

# Restored Image Using Fourier Transform
<img width="400" height="300" alt="Nailpolish_restored" src="https://github.com/user-attachments/assets/a1662cd0-ed28-4ff9-bbcc-8767bddbc793" />

## Summary

This section demonstrates:
  * Spatial-domain convolution (Gaussian blur)
  * Frequency-domain deconvolution (Fourier-based restoration)
  * Practical limitations of inverse filtering (noise amplification, imperfect recovery)
Together, these results illustrate how convolution in the spatial domain corresponds to multiplication in the frequency domain, and how Fourier methods can be used to approximate the inverse of a blur operation.
