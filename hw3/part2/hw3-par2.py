import os
import glob
import cv2
import numpy as np

# =CONFIG 
DATASET_DIR = "part2/hw3-part2-dataset"   # input images
OUT_DIR     = "part2/output_part2"        
os.makedirs(OUT_DIR, exist_ok=True)

# Edge detector parameters
EDGE_BLUR_KSIZE      = (5, 5)   # Gaussian blur for edges
EDGE_SOBEL_KSIZE     = 3       # Sobel kernel size
EDGE_MAG_THRESH_RATIO = 0.50   # stronger threshold ⇒ fewer edge points

# Corner detector parameters (Harris)
CORNER_BLUR_KSIZE     = (7, 7)  # heavier blur to suppress texture noise
CORNER_BLOCK_SIZE     = 5       # neighborhood size in cornerHarris
CORNER_KSIZE          = 5       # Sobel aperture size inside Harris
CORNER_K              = 0.04    # Harris k parameter
CORNER_THRESH_RATIO   = 0.07    # threshold on Harris response
CORNER_MAX_KP         = 200     # max # of corners kept per image


# EDGE KEYPOINTS 
def detect_edge_keypoints(gray,
                          blur_ksize=EDGE_BLUR_KSIZE,
                          sobel_ksize=EDGE_SOBEL_KSIZE,
                          mag_thresh_ratio=EDGE_MAG_THRESH_RATIO):
    """
    Simple EDGE keypoint detector.

    Steps:
      1. Smooth image with Gaussian blur.
      2. Compute Sobel gradients gx, gy.
      3. Compute gradient magnitude M = sqrt(gx^2 + gy^2).
      4. Keep pixels where M is above a global threshold.
      5. Keep only local maxima in a 3x3 neighborhood.
    """

    # 1) smooth to reduce noise
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 1.2)

    # 2) Sobel gradients (float32)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

    # 3) gradient magnitude
    mag = cv2.magnitude(gx, gy)

    # handle blank images
    mag_max = mag.max()
    if mag_max < 1e-6:
        return []

    # 4) threshold on magnitude (strong edges only)
    thresh = mag_thresh_ratio * mag_max
    strong = (mag > thresh).astype(np.uint8)

    # 5) local maxima in 3x3 neighborhood (non-maximum suppression)
    mag_dilated = cv2.dilate(mag, np.ones((3, 3), np.uint8))
    local_max = (mag == mag_dilated).astype(np.uint8)

    edge_mask = strong * local_max  # pixels that are strong AND local maxima

    # collect coordinates
    ys, xs = np.where(edge_mask > 0)
    keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in zip(xs, ys)]

    # optional: downsample if there are too many (for drawing only)
    MAX_EDGES = 500
    if len(keypoints) > MAX_EDGES:
        step = len(keypoints) // MAX_EDGES + 1
        keypoints = keypoints[::step]

    return keypoints


# CORNER KEYPOINTS 
def detect_corner_keypoints(gray,
                            blur_ksize=CORNER_BLUR_KSIZE,
                            block_size=CORNER_BLOCK_SIZE,
                            ksize=CORNER_KSIZE,
                            k=CORNER_K,
                            harris_thresh_ratio=CORNER_THRESH_RATIO,
                            max_kp=CORNER_MAX_KP):
    """
    Simple CORNER keypoint detector using Harris response.

    Steps:
      1. Smooth image with a stronger Gaussian blur.
      2. Run cv2.cornerHarris to get corner response R(x,y).
      3. Threshold R > harris_thresh_ratio * max(R).
      4. Keep only local maxima in a 3x3 neighborhood.
    """

    # 1) blur to suppress small texture noise
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 2.0)

    # 2) Harris corner response
    harris = cv2.cornerHarris(
        gray_blur,
        blockSize=block_size,
        ksize=ksize,
        k=k
    )

    h_max = harris.max()
    if h_max < 1e-6:
        return []

    # 3) threshold on raw Harris response
    thresh = harris_thresh_ratio * h_max
    strong = (harris > thresh)

    # 4) local maxima in 3x3
    h_dilated = cv2.dilate(harris, np.ones((3, 3), np.uint8))
    local_max = (harris == h_dilated)

    corner_mask = strong & local_max

    ys, xs = np.where(corner_mask)
    keypoints = [cv2.KeyPoint(float(x), float(y), 7) for x, y in zip(xs, ys)]

    # limit number of corners for visualization
    if len(keypoints) > max_kp:
        step = len(keypoints) // max_kp + 1
        keypoints = keypoints[::step]

    return keypoints


# MAIN PIPELINE 
def main():
    # collect all images in the dataset folder
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.*")))
    if not img_paths:
        print("No images found in", DATASET_DIR)
        return

    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        print("Processing:", fname)

        # read image
        bgr = cv2.imread(path)
        if bgr is None:
            print("  Could not read", path)
            continue

        # convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # EDGE KEYPOINTS 
        edge_kp = detect_edge_keypoints(gray)
        vis_edge = cv2.drawKeypoints(
            bgr, edge_kp, None,
            color=(0, 0, 255),  # red (edges)
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_edge_kp.png"), vis_edge)
        print(f"  Edge keypoints:   {len(edge_kp)}")

        #  CORNER KEYPOINTS 
        corner_kp = detect_corner_keypoints(gray)
        vis_corner = cv2.drawKeypoints(
            bgr, corner_kp, None,
            color=(0, 255, 0),  # green (corners)
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_corner_kp.png"), vis_corner)
        print(f"  Corner keypoints: {len(corner_kp)}")

    print("Done. Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
