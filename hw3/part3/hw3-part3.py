import os
import glob
import cv2
import numpy as np

# CONFIG 
DATASET_DIR = "part3/hw3-part3-dataset"      
OUT_DIR     = "part3/output_part3"     
os.makedirs(OUT_DIR, exist_ok=True)


def get_object_boundary(bgr):
    """
    Find the object (brain) boundary using classic image processing:

      1) Convert to gray and blur (noise reduction)
      2) Otsu threshold (foreground vs background)
      3) Morphological closing + opening (smooth mask)
      4) Find largest contour = object
      5) Create binary mask + overlay with red contour
    """

    # 1) gray + blur
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Otsu threshold.

    _, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3) Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 4) Find contours and pick the largest one (the brain)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None, None

    main_cnt = max(contours, key=cv2.contourArea)

    # 5a) Binary mask of the object (inside boundary = 255)
    obj_mask = np.zeros_like(mask)
    cv2.drawContours(obj_mask, [main_cnt], -1, 255, thickness=-1)

    # 5b) Overlay red contour on original image
    overlay = bgr.copy()
    cv2.drawContours(overlay, [main_cnt], -1, (0, 0, 255), thickness=2)

    return obj_mask, overlay


#  MAIN 
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
            print("  Could not read image, skipping:", path)
            continue

        obj_mask, overlay = get_object_boundary(bgr)
        if obj_mask is None:
            print("  No object found, skipping:", fname)
            continue

        # Save binary mask and overlay with red boundary
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_mask.png"), obj_mask)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_boundary.png"), overlay)

    print("Done. Check outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
