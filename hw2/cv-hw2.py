import cv2
import numpy as np
import os
import glob

TEMPLATE_DIR = "templates"
SCENE_DIR = "scenes"
RESULTS_DIR = "Downloads/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def autocrop(img, tol=5):
    """
    Automatically crop away almost-empty background (near-black/near-white).
    This makes the detector focus on the object itself.
    """
    if img is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # keep pixels that are not too dark AND not too bright
    mask = (gray > tol) & (gray < 255 - tol)
    if not mask.any():
        return img

    coords = np.argwhere(mask)
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    return img[y1:y2 + 1, x1:x2 + 1]


def get_detector():
    """
    Try to use SIFT (better), fall back to ORB if SIFT is not available.
    Returns (detector, norm_type, name).
    """
    # Try SIFT
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=2500)
        return sift, cv2.NORM_L2, "SIFT"

    # Fallback: ORB
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.1,
        nlevels=12,
        edgeThreshold=10,
        patchSize=31,
        fastThreshold=5,
    )
    return orb, cv2.NORM_HAMMING, "ORB"


def detect_object(scene_path, template_path,
                  ratio_thresh=0.75,
                  min_good_matches=10,
                  min_inliers=8):

    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if scene is None or template is None:
        print(f"[ERROR] Could not read: {scene_path} or {template_path}")
        return

    # Crop template to remove blank border
    template = autocrop(template)

    # Convert to gray
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Contrast normalization (helps with lighting differences)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    scene_gray = clahe.apply(scene_gray)

    # --- feature detector ---
    detector, norm_type, det_name = get_detector()

    kp1, des1 = detector.detectAndCompute(template_gray, None)
    kp2, des2 = detector.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        print(f"[WARN] No descriptors / too few keypoints for {os.path.basename(template_path)}")
        result = scene.copy()
        label = f"{os.path.splitext(os.path.basename(template_path))[0]}: Not detected"
        cv2.putText(result, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out_path = os.path.join(
            RESULTS_DIR,
            f"orb_{os.path.basename(template_path)}_{os.path.basename(scene_path)}"
        )
        cv2.imwrite(out_path, result)
        return

    # --- matching with Lowe ratio test ---
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    print(f"{det_name}: {os.path.basename(template_path)} in {os.path.basename(scene_path)} "
          f"— good matches: {len(good)}")

    result = scene.copy()
    obj_name = os.path.splitext(os.path.basename(template_path))[0]

    if len(good) < min_good_matches:
        # not enough matches to trust homography
        text = f"{obj_name}: Not enough matches ({len(good)}/{min_good_matches})"
        cv2.putText(result, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # build point arrays
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # robust homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        if H is not None and mask is not None:
            inliers = int(mask.sum())
            print(f"  inliers: {inliers}/{len(good)}")

            if inliers >= min_inliers:
                h, w = template_gray.shape
                corners = np.float32([
                    [0, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [w - 1, 0]
                ]).reshape(-1, 1, 2)

                # project template corners into scene
                projected = cv2.perspectiveTransform(corners, H)

                # ---- RED RECTANGULAR BOUNDING BOX ----
                xs = projected[:, 0, 0]
                ys = projected[:, 0, 1]
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())

                # clamp to image bounds
                h_scene, w_scene = result.shape[:2]
                x_min = max(0, min(x_min, w_scene - 1))
                x_max = max(0, min(x_max, w_scene - 1))
                y_min = max(0, min(y_min, h_scene - 1))
                y_max = max(0, min(y_max, h_scene - 1))

                # draw red rectangle
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max),
                              (0, 0, 255), 4)

                label = f"{obj_name} detected"
                cv2.putText(result, label, (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                text = f"{obj_name}: Low inliers ({inliers}/{len(good)})"
                cv2.putText(result, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(result, f"{obj_name}: Homography failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out_path = os.path.join(
        RESULTS_DIR,
        f"orb_{os.path.basename(template_path)}_{os.path.basename(scene_path)}"
    )
    cv2.imwrite(out_path, result)


def main():
    templates = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*")))
    scenes = sorted(glob.glob(os.path.join(SCENE_DIR, "*")))

    print(f"Found {len(templates)} templates and {len(scenes)} scenes.")
    if not templates or not scenes:
        print("No images found in templates/ or scenes/")
        return

    for t in templates:
        for s in scenes:
            detect_object(s, t)

    print("DONE. Check 'results/' for outputs.")


if __name__ == "__main__":
    main()
