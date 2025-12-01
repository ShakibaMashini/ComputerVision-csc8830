import cv2
import numpy as np
import json

# 1. INPUT STEREO IMAGES
# Replace with your own file names
left_path  = "left.jpg"
right_path = "right.jpg"

left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

if left is None or right is None:
    raise FileNotFoundError("Could not load left/right images")

if left.shape != right.shape:
    raise ValueError("Left and right images must have same size")


#  2. STEREO MATCHING (DISPARITY)
# Parameters: you can tune later, but this is a good starting point
min_disparity = 0
num_disparities = 64   # must be divisible by 16
block_size = 7    # must be odd

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# OpenCV returns disparity in fixed-point format (scaled by 16)
disp_raw = stereo.compute(left, right).astype(np.float32)

# Convert to "real" disparity in pixels
disp = disp_raw / 16.0

# Optional: mask invalid disparities (<=0)
disp[disp <= 0] = np.nan


#3. SAVE DISPARITY MAP FOR WEB USE 
# For the web, we need to store it as an 8-bit image.
# We'll normalize disparity to [0, 255] just for visualization/storage.
valid_disp = disp[np.isfinite(disp)]
min_valid = valid_disp.min()
max_valid = valid_disp.max()

print("Disparity range (valid):", min_valid, "to", max_valid)

# Normalize to 0-255 range
disp_norm = (disp - min_valid) / (max_valid - min_valid)
disp_norm = np.clip(disp_norm, 0, 1)
disp_8u = (disp_norm * 255).astype(np.uint8)

cv2.imwrite("disparity.png", disp_8u)
print("Saved disparity.png")

# IMPORTANT: We must remember how to go back from disp_8u to true disparity.
# If disp_8u = 0..255, then:
#   disp = min_valid + (disp_8u / 255) * (max_valid - min_valid)


#  4. SAVE CALIBRATION + NORMALIZATION PARAMS
# Fill these with real values from your stereo calibration
fx = 800.0   # example focal length in pixels
fy = 800.0   # if different, we will use fx for depth formula
cx = left.shape[1] / 2.0
cy = left.shape[0] / 2.0
baseline = 0.1  # e.g. 0.1 meters = 10cm

params = {
    "fx": float(fx),
    "fy": float(fy),
    "cx": float(cx),
    "cy": float(cy),
    "baseline": float(baseline),
    "disp_min_valid": float(min_valid),
    "disp_max_valid": float(max_valid)
}

with open("stereo_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Saved stereo_params.json")
