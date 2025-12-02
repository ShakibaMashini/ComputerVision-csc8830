import os
import glob
import cv2
import numpy as np

from ultralytics import SAM

# -------- PATHS --------
DATASET_DIR   = "part4/dataset"      # your input images
SAM2_MASK_DIR = "part4/sam2_masks"   # where we will save SAM2 masks

os.makedirs(SAM2_MASK_DIR, exist_ok=True)


def build_sam2_model():
    """
    Build a SAM-2 model from Ultralytics.

    You can change 'sam2_b.pt' to 'sam2_t.pt', 'sam2_s.pt', etc.,
    depending on what you downloaded / want to use.
    """
    # This will download sam2_b.pt the first time if not present
    model = SAM("sam2_b.pt")
    return model


def run_sam2_on_image(model, bgr_img):
    """
    Run SAM-2 on a single image using Ultralytics.

    We let the model autosegment the entire image (no manual prompt)
    and then take the largest mask as the object of interest.

    Returns:
        mask: (H, W) uint8 in {0, 255}
    """
    # BGR -> RGB for visualization consistency (Ultralytics accepts either),
    # but we will just pass the BGR np.ndarray directly; Ultralytics can handle it.
    img = bgr_img.copy()

    # Run model.predict on this single image
    # We use retina_masks=True for higher-quality masks
    results = model.predict(source=img, retina_masks=True, verbose=False)

    if not results:
        # no segmentation result
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    r = results[0]  # first (and only) image result

    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        # no masks found
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    # r.masks.data is a tensor of shape (N, H, W) with values in [0,1]
    masks = r.masks.data.cpu().numpy()  # (N, H, W)

    # Choose the mask with the largest area as "the object"
    areas = masks.sum(axis=(1, 2))  # sum of pixel probs per mask
    best_idx = int(np.argmax(areas))
    best_mask = masks[best_idx]  # (H, W), float in [0,1]

    # Binarize with threshold 0.5
    binary = (best_mask > 0.5).astype(np.uint8) * 255  # {0,255}
    return binary


def main():
    model = build_sam2_model()

    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.*")))
    if not img_paths:
        print("No images found in", DATASET_DIR)
        return

    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        print("Processing:", fname)

        bgr = cv2.imread(path)
        if bgr is None:
            print("  Could not read image:", path)
            continue

        try:
            mask = run_sam2_on_image(model, bgr)
        except Exception as e:
            print("  SAM-2 failed on", fname, "->", repr(e))
            continue

        out_path = os.path.join(SAM2_MASK_DIR, fname + ".png")
        cv2.imwrite(out_path, mask)
        print("  Saved SAM-2 mask to:", out_path)

    print("\nDone. SAM-2 masks are in:", SAM2_MASK_DIR)


if __name__ == "__main__":
    main()
