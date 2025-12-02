import os
import glob
import cv2
import numpy as np

# Paths you can change depending on your files
IMAGE_DIR = "images"                 
OUTPUT_PANO = "output_panorama.png"
PHONE_PANO = "phone_panorama.jpg"    
OUTPUT_COMPARISON = "comparison_with_phone_panorama.png"

# Resize factor for faster stitching (1.0 = original size)
RESIZE_SCALE = 0.7


def load_images_from_folder(folder, scale=1.0):
    """
    Loads all images from the given folder and returns them as a list.
    The images are sorted by filename so the stitching follows left→right order.
    You can also downscale the images to speed things up.
    """
    image_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    images = []

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue

        if scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        images.append(img)

    return images


def create_feature_detector():
    """
    Creates a SIFT feature detector if installed.
    If SIFT is not available, it automatically switches to ORB.
    """
    if hasattr(cv2, "SIFT_create"):
        print("[INFO] Using SIFT features")
        return cv2.SIFT_create(), "sift"
    else:
        print("[INFO] SIFT not available. Using ORB instead.")
        return cv2.ORB_create(5000), "orb"


def detect_and_describe(image, detector):
    """
    Runs feature detection and descriptor extraction on an image.
    Returns:
      - keypoints
      - descriptor vectors
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(descA, descB, method="sift", ratio=0.75):
    """
    Matches features between two images using KNN + Lowe's ratio test.
    SIFT uses L2 distance, ORB uses Hamming distance.
    """
    if descA is None or descB is None:
        return []

    if method == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw_matches = matcher.knnMatch(descA, descB, k=2)
    good_matches = []

    # Lowe's ratio filtering to keep only the best matches
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def compute_homography(kpsA, kpsB, matches, reproj_thresh=4.0):
    """
    Computes a homography between two images using matched keypoints.
    RANSAC is used to handle outliers and produce a robust estimate.
    """
    if len(matches) < 4:
        return None, None

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reproj_thresh)
    return H, status


def warp_and_blend(base, new_img, H):
    """
    Warps the new image into the coordinate system of the current panorama
    using the homography matrix. Then it blends them together on a larger canvas.
    """
    h1, w1 = base.shape[:2]
    h2, w2 = new_img.shape[:2]

    # Compute image corners so we can determine the final canvas size
    base_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    new_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Project corners of the new image into the base image
    new_corners_warped = cv2.perspectiveTransform(new_corners, H)

    # Combine corners to figure out panorama boundaries
    all_corners = np.concatenate((base_corners, new_corners_warped), axis=0)
    xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation shift so that all coordinates are positive
    tx, ty = -xmin, -ymin
    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    pano_width = xmax - xmin
    pano_height = ymax - ymin

    # Warp the new image into the panorama canvas
    panorama = cv2.warpPerspective(new_img, translation @ H, (pano_width, pano_height))

    # Place the existing panorama at its corresponding location in the new canvas
    panorama[ty:ty + h1, tx:tx + w1] = base

    # This is a simple merge; more advanced blending is possible if needed
    return panorama


def stitch_sequence(images, ratio=0.75, reproj_thresh=4.0):
    """
    Stitches a list of images into a panorama.
    We start with the first image and keep adding the next images one by one.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")

    detector, method = create_feature_detector()
    panorama = images[0].copy()

    for idx in range(1, len(images)):
        print(f"[INFO] Stitching image {idx + 1} of {len(images)}")

        img = images[idx]

        # Extract features from both the current panorama and the new image
        kpsA, descA = detect_and_describe(panorama, detector)
        kpsB, descB = detect_and_describe(img, detector)

        print(f"  - keypoints in panorama: {len(kpsA)}, in new image: {len(kpsB)}")

        matches = match_features(descA, descB, method=method, ratio=ratio)
        print(f"  - matches kept after ratio test: {len(matches)}")

        if len(matches) < 4:
            print("[WARN] Not enough matches. Skipping this image.")
            continue

        # Compute homography using RANSAC
        H, status = compute_homography(kpsA, kpsB, matches, reproj_thresh=reproj_thresh)
        if H is None:
            print("[WARN] Could not compute homography. Skipping this image.")
            continue

        # Warp and merge the new image into the panorama
        panorama = warp_and_blend(panorama, img, H)

    return panorama


def create_side_by_side(our_pano, phone_pano_path, output_path):
    """
    If a phone-generated panorama exists, load it and create a side-by-side
    image comparing it with the stitched panorama from this script.
    """
    if not os.path.exists(phone_pano_path):
        print("[INFO] Phone panorama not found. Skipping comparison.")
        return

    phone = cv2.imread(phone_pano_path, cv2.IMREAD_COLOR)
    if phone is None:
        print("[INFO] Unable to read phone panorama. Skipping.")
        return

    # Match the phone panorama's height to our stitched result
    h_ours, _ = our_pano.shape[:2]
    h_phone, w_phone = phone.shape[:2]

    scale = h_ours / float(h_phone)
    phone_resized = cv2.resize(phone, (int(w_phone * scale), h_ours), interpolation=cv2.INTER_AREA)

    # Combine them horizontally
    comparison = cv2.hconcat([our_pano, phone_resized])
    cv2.imwrite(output_path, comparison)

    print(f"[INFO] Saved comparison image to: {output_path}")


def main():
    images = load_images_from_folder(IMAGE_DIR, scale=RESIZE_SCALE)
    print(f"[INFO] Loaded {len(images)} images from '{IMAGE_DIR}'")

    if len(images) < 2:
        print("[ERROR] Not enough images to build a panorama.")
        return

    panorama = stitch_sequence(images)
    cv2.imwrite(OUTPUT_PANO, panorama)
    print(f"[INFO] Panorama saved to: {OUTPUT_PANO}")

    create_side_by_side(panorama, PHONE_PANO, OUTPUT_COMPARISON)


if __name__ == "__main__":
    main()
