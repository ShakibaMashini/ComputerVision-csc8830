import os
import glob
import math
import cv2
import numpy as np

#  CONFIG 
DATASET_DIR = "dataset"                    
OUT_DIR     = "output_aruco_seg"      
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate dictionaries to try (we don't know which one you printed)
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
    Try multiple dictionaries and return the detection with
    the largest number of markers.

    Returns:
        best_corners, best_ids, best_dict_name
        (corners is a list of (1,4,2) arrays, ids shape is (N,1))
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


#MARKER SIZE COMPUTATION 
def compute_marker_sizes(corners, ids):
    """
    For each marker, compute average side length (px),
    area (px^2), and marker center.

    Returns a list of dicts:
        {
            "id": marker_id,
            "side": avg_side_px,
            "area": area_px2,
            "center": (cx, cy),
        }
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


# SEGMENTATION
def segment_object_with_aruco(bgr):
    """
    Segment non-rectangular object using ArUco markers on its boundary.

    Returns:
        markers_vis: image with markers drawn
        obj_mask:    binary mask (255 inside object)
        overlay:     original + red boundary
        marker_sizes: list from compute_marker_sizes()
        ids:         marker IDs
        dict_name:   dictionary name that worked
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Optional: light denoising
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    corners, ids, dict_name = detect_aruco_any(gray_blur)

    if ids is None or len(corners) == 0:
        return None, None, None, None, None, None

    # Show detected markers
    # (needs the dictionary we ended up using)
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)
    else:
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)

    # Collect all corners and compute convex hull
    pts_list = []
    for c in corners:
        pts_list.append(c.reshape(-1, 2))
    pts_all = np.vstack(pts_list).astype(np.float32)
    hull = cv2.convexHull(pts_all)

    # Mask
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(obj_mask, hull.astype(np.int32), 255)

    # Overlay with red boundary
    overlay = bgr.copy()
    cv2.drawContours(overlay, [hull.astype(np.int32)], -1, (0, 0, 255), 3)

    # Marker sizes
    marker_sizes = compute_marker_sizes(corners, ids)

    return markers_vis, obj_mask, overlay, marker_sizes, ids, dict_name


# MAIN
def main():
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

        (markers_vis,
         mask,
         overlay,
         marker_sizes,
         ids,
         dict_name) = segment_object_with_aruco(bgr)

        if mask is None:
            print("  No ArUco markers detected in:", fname)
            continue

        print(f"  Used dictionary: {dict_name}")
        print("  Detected markers and sizes:")
        for m in marker_sizes:
            print(f"    id={m['id']:2d}  side≈{m['side']:.2f} px   area≈{m['area']:.2f} px^2")

        # Draw size text on top of overlay
        overlay_with_size = overlay.copy()
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

        # Save
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_markers.png"),  markers_vis)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_mask.png"),     mask)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_boundary.png"), overlay)
        cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_size.png"),     overlay_with_size)

    print("Done. Check outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
