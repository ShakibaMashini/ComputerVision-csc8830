import os
import glob
import cv2
import numpy as np

# CONFIG 
DATASET_DIR   = "part5/dataset"             # input images
OUT_DIR       = "part5/output_aruco_seg_2"  
SAM2_MASK_DIR = "part5/sam2_masks"          # SAM2 mask images (binary)

os.makedirs(OUT_DIR, exist_ok=True)

# Candidate dictionaries 
CANDIDATE_DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]

# ARUCO DETECTION

def detect_aruco_any(gray):
    """
    Try multiple ArUco / AprilTag dictionaries and return
    the detection with the largest number of markers.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV build has no cv2.aruco module.")

    best_corners, best_ids, best_dict_name = [], None, None
    max_markers = 0

    for name in CANDIDATE_DICT_NAMES:
        # Skip dicts that don't exist in this OpenCV build
        if not hasattr(cv2.aruco, name):
            continue

        dict_id = getattr(cv2.aruco, name)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # New vs old API
        if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        num = 0 if ids is None else len(ids)
        if num > max_markers:
            max_markers = num
            best_corners, best_ids, best_dict_name = corners, ids, name

    return best_corners, best_ids, best_dict_name



# MARKER SIZE COMPUTATION

def compute_marker_sizes(corners, ids):
    """
    For each marker, compute:
      - average side length (px)
      - area (px^2)
      - center (cx, cy)
    """
    sizes = []
    if corners is None or len(corners) == 0:
        return sizes

    for i, c in enumerate(corners):
        pts = c.reshape(-1, 2).astype(np.float32)  # (4,2)

        # side lengths between consecutive corners
        side_lengths = []
        for j in range(4):
            p1 = pts[j]
            p2 = pts[(j + 1) % 4]
            side_lengths.append(np.linalg.norm(p1 - p2))
        avg_side = float(np.mean(side_lengths))

        area = float(cv2.contourArea(pts))
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        marker_id = int(ids[i][0]) if ids is not None else -1

        sizes.append({
            "id": marker_id,
            "side": avg_side,
            "area": area,
            "center": (cx, cy),
        })

    return sizes



#ARUCO-BASED SEGMENTATION

def segment_object_with_aruco(bgr):
    """
    Segment non-rectangular object using ArUco markers on its boundary.

    Returns:
        markers_vis : image with markers drawn
        obj_mask    : binary mask (255 inside object)
        overlay     : original + red boundary (our segmentation)
        marker_sizes
        ids
        dict_name
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # light denoising
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    corners, ids, dict_name = detect_aruco_any(gray_blur)

    if ids is None or len(corners) == 0:
        return None, None, None, None, None, None

    # draw detected markers
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)
    else:
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)

    # collect all marker corners and compute convex hull
    pts_list = [c.reshape(-1, 2) for c in corners]
    pts_all = np.vstack(pts_list).astype(np.float32)
    hull = cv2.convexHull(pts_all)

    # binary mask from hull
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(obj_mask, hull.astype(np.int32), 255)

    # overlay: red boundary from our hull
    overlay = bgr.copy()
    cv2.drawContours(overlay, [hull.astype(np.int32)], -1, (0, 0, 255), 3)

    marker_sizes = compute_marker_sizes(corners, ids)
    return markers_vis, obj_mask, overlay, marker_sizes, ids, dict_name



# SAM2 MASK LOADING + COMPARISON

def load_sam2_mask_for_image(img_path):
    """
    Load the SAM2 segmentation mask corresponding to this image.

    We try these patterns inside SAM2_MASK_DIR:
      1) <base>_mask.(png|jpg|jpeg)
      2) <base>.(png|jpg|jpeg)
    """
    base = os.path.splitext(os.path.basename(img_path))[0]

    candidates = []
    # prefer *_mask.*
    for ext in ("png", "jpg", "jpeg"):
        candidates.append(os.path.join(SAM2_MASK_DIR, f"{base}_mask.{ext}"))
    # then plain base.*
    for ext in ("png", "jpg", "jpeg"):
        candidates.append(os.path.join(SAM2_MASK_DIR, f"{base}.{ext}"))

    for path in candidates:
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask

    # last resort: any file starting with base
    wildcard = os.path.join(SAM2_MASK_DIR, base + "*")
    for path in glob.glob(wildcard):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask

    return None


def compute_iou_and_dice(mask_a, mask_b):
    """
    Compute IoU and Dice between two binary masks (0/255).
    """
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()

    if union == 0:
        iou = 0.0
    else:
        iou = inter / float(union)

    if a_sum + b_sum == 0:
        dice = 0.0
    else:
        dice = 2.0 * inter / float(a_sum + b_sum)

    return iou, dice


def make_comparison_overlay(bgr, mask_aruco, mask_sam2):
    """
    Show both boundaries on one image:
      - red   = ArUco segmentation boundary
      - green = SAM2 segmentation boundary
    """
    h, w = bgr.shape[:2]

    # ensure same size
    mask_sam2_resized = cv2.resize(mask_sam2, (w, h), interpolation=cv2.INTER_NEAREST)

    # find contours for drawing
    contours_aruco, _ = cv2.findContours(
        (mask_aruco > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_sam2, _ = cv2.findContours(
        (mask_sam2_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = bgr.copy()
    # red: our ArUco result
    cv2.drawContours(overlay, contours_aruco, -1, (0, 0, 255), 2)
    # green: SAM2
    cv2.drawContours(overlay, contours_sam2, -1, (0, 255, 0), 2)

    return overlay, mask_sam2_resized


#  MAIN

def main():
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.*")))
    if not img_paths:
        print("No images found in", DATASET_DIR)
        return

    all_metrics = []

    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        print("\nProcessing:", fname)

        bgr = cv2.imread(path)
        if bgr is None:
            print("  Could not read image:", path)
            continue

        # ArUco segmentation
        (markers_vis,
         mask_aruco,
         overlay_aruco,
         marker_sizes,
         ids,
         dict_name) = segment_object_with_aruco(bgr)

        if mask_aruco is None:
            print("  No ArUco markers detected in:", fname)
            continue

        print(f"  Used dictionary: {dict_name}")
        print("  Detected markers and sizes:")
        for m in marker_sizes:
            print(f"    id={m['id']:2d}  side≈{m['side']:.2f} px   area≈{m['area']:.2f} px^2")

        # annotate ArUco sizes on overlay
        overlay_with_size = overlay_aruco.copy()
        for m in marker_sizes:
            cx, cy = m["center"]
            text = f"{m['side']:.1f}px"
            cv2.putText(
                overlay_with_size,
                text,
                (int(cx) - 20, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        # save ArUco outputs
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_markers.png"),  markers_vis)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_mask_aruco.png"), mask_aruco)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_boundary_aruco.png"), overlay_aruco)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_size_aruco.png"), overlay_with_size)

        # SAM2 comparison 
        sam2_mask = load_sam2_mask_for_image(path)
        if sam2_mask is None:
            print("  No SAM2 mask found for", fname, "→ skipping comparison.")
            continue

        comp_overlay, sam2_mask_resized = make_comparison_overlay(
            bgr, mask_aruco, sam2_mask
        )

        iou, dice = compute_iou_and_dice(mask_aruco, sam2_mask_resized)
        print(f"  SAM2 vs ArUco: IoU = {iou:.3f}, Dice = {dice:.3f}")

        # save comparison visuals
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_mask_sam2.png"), sam2_mask_resized)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_compare_overlay.png"), comp_overlay)

        all_metrics.append((fname, iou, dice))

    # summary
    if all_metrics:
        print("\n=== Summary (ArUco vs SAM2) ===")
        for name, iou, dice in all_metrics:
            print(f"{name}: IoU={iou:.3f}, Dice={dice:.3f}")

    print("\nDone. Check outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
