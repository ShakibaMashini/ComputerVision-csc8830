import os
import glob
import cv2
import numpy as np

DATASET_DIR = "part1/hw3_dataset"        
OUT_DIR = "part1/output_task1"            
os.makedirs(OUT_DIR, exist_ok=True)

def normalize_to_uint8(img):
    """
    Normalize arbitrary float / large-range image to 0–255 uint8 for saving.
    """
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def compute_gradient_and_log(gray):
    """
    Input:  gray image (uint8)
    Output:
      grad_mag_u8  – gradient magnitude (uint8)
      grad_ang_u8  – gradient angle (uint8, encoded for visualization)
      log_u8       – Laplacian-of-Gaussian response (uint8)
    """

    #  Gradients via Sobel 
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # magnitude and angle
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)   # radians, range [-pi, pi]

    # normalize magnitude to [0, 255]
    grad_mag_u8 = normalize_to_uint8(mag)

    # convert angle to degrees in [0,180] and normalize to [0,255] for display
    ang_deg = np.degrees(ang)
    ang_deg[ang_deg < 0] += 180.0
    grad_ang_u8 = normalize_to_uint8(ang_deg)

    # Laplacian of Gaussian 
    # 1) Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0, sigmaY=1.0)
    # 2) Laplacian filter
    log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_u8 = normalize_to_uint8(log)

    return grad_mag_u8, grad_ang_u8, log_u8

def main():
    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.*")))
    if not image_paths:
        print("No images found in", DATASET_DIR)
        return

    for path in image_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        print("Processing:", fname)

        bgr = cv2.imread(path)
        if bgr is None:
            print("  Could not read image, skipping.")
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        grad_mag_u8, grad_ang_u8, log_u8 = compute_gradient_and_log(gray)

        # save results
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_grad_mag.png"), grad_mag_u8)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_grad_ang.png"), grad_ang_u8)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_log.png"),      log_u8)

    print("Done. Check outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
