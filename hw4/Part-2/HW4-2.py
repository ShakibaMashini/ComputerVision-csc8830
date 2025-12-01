import cv2
import numpy as np
import random
import math
import os


# Paths to the two images you want to compare
IMG1_PATH = "data/im1.jpg"
IMG2_PATH = "data/im2.jpg"

OUT_MATCHES_OURS = "ours_matches.png"
OUT_MATCHES_OPENCV = "opencv_matches.png"


# Utility Function
def to_gray_float(img):
    """
    Converts a BGR image to grayscale float32 in the range [0,1].
    This makes the processing easier for Gaussian blurs and gradients.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float32) / 255.0


# SIFT Implementation 

class SIFTFromScratch:
    """
    A simplified, educational SIFT implementation that follows
    the main ideas of the original algorithm but avoids highly
    optimized or complex refinements.
    """

    def __init__(self,
                 num_octaves=4,
                 num_scales=3,
                 sigma=1.6,
                 contrast_thresh=0.04,
                 edge_thresh=10,
                 descriptor_width=4,
                 descriptor_bins=8):

        # Algorithm parameters
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh
        self.desc_width = descriptor_width
        self.desc_bins = descriptor_bins

        # Scale step factor between Gaussian levels
        self.k = 2 ** (1.0 / self.num_scales)

    # Gaussian & DoG Pyramids 
    def build_gaussian_pyramid(self, base):
        """
        Builds a Gaussian pyramid where each octave contains several
        progressively blurred images. Additional scales are included
        to generate DoG extrema.
        """
        gauss_pyr = []
        base = to_gray_float(base)
        current = base.copy()

        num_scales_total = self.num_scales + 3  # Required for DoG stability

        for o in range(self.num_octaves):
            octave_images = []

            # Compute sigmas for this octave
            sigmas = [self.sigma * (self.k ** i) for i in range(num_scales_total)]

            prev = current
            for s in range(num_scales_total):
                if s == 0 and o == 0:
                    # The first image of the entire pyramid gets a direct blur
                    img_blur = cv2.GaussianBlur(prev, (0, 0), sigmas[s])
                elif s == 0:
                    # For later octaves, the base image is already blurred
                    img_blur = prev
                else:
                    # Blur relative to previous scale
                    sigma_diff = math.sqrt(sigmas[s] ** 2 - sigmas[s - 1] ** 2)
                    img_blur = cv2.GaussianBlur(prev, (0, 0), sigma_diff)

                octave_images.append(img_blur)
                prev = img_blur

            gauss_pyr.append(octave_images)

            # Downsample by factor of 2 to create next octave base
            next_base = octave_images[self.num_scales]
            new_size = (next_base.shape[1] // 2, next_base.shape[0] // 2)
            current = cv2.resize(next_base, new_size, interpolation=cv2.INTER_NEAREST)

        return gauss_pyr

    def build_dog_pyramid(self, gauss_pyr):
        """
        Constructs the Difference-of-Gaussians pyramid by subtracting
        adjacent Gaussian-blurred images.
        """
        dog_pyr = []
        for octave in gauss_pyr:
            dogs = []
            for s in range(len(octave) - 1):
                dogs.append(octave[s + 1] - octave[s])
            dog_pyr.append(dogs)
        return dog_pyr

    # Keypoint Detection 

    def detect_keypoints(self, dog_pyr):
        """
        Detects local maxima/minima across space and scale
        in the DoG pyramid.
        """
        keypoints = []

        for o, dogs in enumerate(dog_pyr):
            for s in range(1, len(dogs) - 1):  # skip the edges in scale-space
                dog = dogs[s]
                prev_dog = dogs[s - 1]
                next_dog = dogs[s + 1]

                h, w = dog.shape

                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        val = dog[y, x]

                        # Remove keypoints with very low contrast
                        if abs(val) < self.contrast_thresh:
                            continue

                        # Extract 3×3×3 neighborhood around the pixel
                        patch_prev = prev_dog[y - 1:y + 2, x - 1:x + 2]
                        patch_curr = dog[y - 1:y + 2, x - 1:x + 2]
                        patch_next = next_dog[y - 1:y + 2, x - 1:x + 2]
                        patch_stack = np.stack([patch_prev, patch_curr, patch_next], axis=0)

                        # Check if this is a local extremum
                        if (val > 0 and val >= patch_stack.max()) or \
                           (val < 0 and val <= patch_stack.min()):
                            if not self._is_edge_like(dog, x, y):
                                keypoints.append((o, s, y, x))

        return keypoints

    def _is_edge_like(self, dog_img, x, y):
        """
        Filters out unstable keypoints that lie along edges by checking
        the ratio of eigenvalues of the local Hessian matrix.
        """
        Dxx = dog_img[y, x + 1] + dog_img[y, x - 1] - 2 * dog_img[y, x]
        Dyy = dog_img[y + 1, x] + dog_img[y - 1, x] - 2 * dog_img[y, x]
        Dxy = (
            dog_img[y + 1, x + 1] - dog_img[y + 1, x - 1]
            - dog_img[y - 1, x + 1] + dog_img[y - 1, x - 1]
        ) / 4.0

        tr = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        if det <= 0:
            return True

        r = self.edge_thresh
        if (tr * tr) / det > ((r + 1) ** 2) / r:
            return True

        return False

    # -------------------- Orientation Assignment -------------------

    def assign_orientations(self, keypoints, gauss_pyr):
        """
        Assigns a dominant orientation to every keypoint using its local gradient
        distribution. This makes descriptors rotation-invariant.
        """
        oriented_kps = []

        for (o, s, y, x) in keypoints:
            gauss = gauss_pyr[o][s + 1]

            sigma = 1.5 * self.sigma
            radius = int(round(3 * sigma))

            h, w = gauss.shape
            hist = np.zeros(36, dtype=np.float32)  # 10-degree bins

            # Accumulate weighted gradient magnitudes into orientation bins
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = y + dy
                    xx = x + dx
                    if yy <= 0 or yy >= h - 1 or xx <= 0 or xx >= w - 1:
                        continue

                    mag, theta = self._grad_mag_ori(gauss, xx, yy)
                    weight = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                    bin_idx = int(theta // 10) % 36
                    hist[bin_idx] += weight * mag

            # Slight smoothing of orientation histogram
            hist = self._smooth_hist(hist)

            # Pick the dominant orientation
            if hist.max() > 1e-6:
                main_orientation = np.argmax(hist) * 10.0
                oriented_kps.append((o, s, y, x, main_orientation))

        return oriented_kps

    def _grad_mag_ori(self, img, x, y):
        """
        Computes gradient magnitude and orientation using central differences.
        """
        dx = img[y, x + 1] - img[y, x - 1]
        dy = img[y - 1, x] - img[y + 1, x]
        magnitude = math.sqrt(dx * dx + dy * dy)
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        return magnitude, angle

    def _smooth_hist(self, hist):
        """
        Smooths a circular histogram by averaging neighboring bins.
        """
        smoothed = np.zeros_like(hist)
        n = len(hist)
        for i in range(n):
            smoothed[i] = (
                hist[i - 2] + hist[i - 1] + hist[i]
                + hist[(i + 1) % n] + hist[(i + 2) % n]
            ) / 5.0
        return smoothed

    # Descriptor Computation 

    def compute_descriptors(self, oriented_kps, gauss_pyr):
        """
        Builds the 128-dimensional SIFT descriptors by dividing the keypoint
        region into a 4×4 grid and computing gradient orientation histograms
        for each cell.
        """
        descriptors = []
        keypoints_xy = []

        for (o, s, y, x, angle) in oriented_kps:
            gauss = gauss_pyr[o][s + 1]
            h, w = gauss.shape

            win_size = 16
            half = win_size // 2

            cos_t = math.cos(math.radians(angle))
            sin_t = math.sin(math.radians(angle))

            hist = np.zeros((self.desc_width, self.desc_width, self.desc_bins), dtype=np.float32)

            sigma_desc = win_size / 2.0
            exp_factor = -1.0 / (2 * sigma_desc * sigma_desc)

            # Iterate over a 16×16 region around the keypoint
            for iy in range(-half, half):
                for ix in range(-half, half):

                    # Rotate window coordinates
                    rx = ix * cos_t - iy * sin_t
                    ry = ix * sin_t + iy * cos_t

                    yy = int(round(y + ry))
                    xx = int(round(x + rx))

                    if yy <= 0 or yy >= h - 1 or xx <= 0 or xx >= w - 1:
                        continue

                    mag, theta = self._grad_mag_ori(gauss, xx, yy)
                    rel_theta = (theta - angle + 360.0) % 360.0

                    weight = math.exp((rx * rx + ry * ry) * exp_factor)

                    # Determine which cell of the 4×4 descriptor grid
                    cx = int((rx + half) / (win_size / self.desc_width))
                    cy = int((ry + half) / (win_size / self.desc_width))

                    if 0 <= cx < self.desc_width and 0 <= cy < self.desc_width:
                        bin_idx = int(rel_theta // (360.0 / self.desc_bins)) % self.desc_bins
                        hist[cy, cx, bin_idx] += weight * mag

            # Flatten the histogram into 128D
            desc = hist.flatten()

            # Normalize descriptor
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                desc /= norm

            # Clip large values and renormalize
            desc = np.clip(desc, 0, 0.2)
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                desc /= norm

            # Convert octave coordinates to original image coordinates
            scale_factor = 2 ** o
            x_orig = x * scale_factor
            y_orig = y * scale_factor

            keypoints_xy.append((x_orig, y_orig))
            descriptors.append(desc)

        if len(descriptors) == 0:
            return [], np.empty((0, 128), dtype=np.float32)

        return keypoints_xy, np.vstack(descriptors).astype(np.float32)

    #  Full SIFT Pipeline 

    def detect_and_compute(self, img):
        """
        Runs the entire SIFT pipeline on an image and returns:
          - keypoint coordinates in the original scale
          - descriptors (Nx128)
        """
        gauss_pyr = self.build_gaussian_pyramid(img)
        dog_pyr = self.build_dog_pyramid(gauss_pyr)
        keypoints = self.detect_keypoints(dog_pyr)
        oriented_kps = self.assign_orientations(keypoints, gauss_pyr)
        return self.compute_descriptors(oriented_kps, gauss_pyr)


# ------------------ Descriptor Matching (Lowe Ratio) ---------------------

def match_descriptors(descs1, descs2, ratio=0.75):
    """
    Matches descriptors from two images using Euclidean distance and
    keeps only the matches that pass Lowe’s ratio test.
    """
    if len(descs1) == 0 or len(descs2) == 0:
        return []

    matches = []

    for i, d1 in enumerate(descs1):
        diffs = descs2 - d1
        dists = np.sum(diffs * diffs, axis=1)

        if len(dists) < 2:
            continue

        nn1_idx = np.argmin(dists)
        nn1_dist = dists[nn1_idx]

        dists[nn1_idx] = np.inf
        nn2_idx = np.argmin(dists)
        nn2_dist = dists[nn2_idx]

        if nn1_dist < ratio * nn2_dist:
            matches.append((i, nn1_idx))

    return matches


# RANSAC (From Scratch) 

def compute_homography_from_points(pts1, pts2):
    """
    Computes a homography using the DLT method based on corresponding points.
    """
    if pts1.shape[0] < 4:
        return None

    A = []
    for (x, y), (xp, yp) in zip(pts1, pts2):
        A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H


def ransac_homography(kpts1, kpts2, matches, num_iter=2000, thresh=3.0):
    """
    Estimates a homography using RANSAC by repeatedly sampling subsets
    of correspondences and keeping the model with the most inliers.
    """
    if len(matches) < 4:
        return None, None

    pts1 = np.array([kpts1[i1] for i1, _ in matches])
    pts2 = np.array([kpts2[i2] for _, i2 in matches])

    best_H = None
    best_inliers = None
    best_count = 0

    for _ in range(num_iter):
        idxs = random.sample(range(len(matches)), 4)
        H = compute_homography_from_points(pts1[idxs], pts2[idxs])

        if H is None:
            continue

        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        proj = (H @ pts1_h.T).T
        proj[:, :2] /= proj[:, 2:3]

        errors = np.linalg.norm(pts2 - proj[:, :2], axis=1)
        inliers = errors < thresh
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers


#  Visualization Helper 

def draw_matches(img1, img2, kpts1, kpts2, matches, inliers_mask=None, max_draw=100):
    """
    Draws lines between matched keypoints in two images.
    Inliers appear in green, outliers (if shown) in red.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    for idx, (i1, i2) in enumerate(matches[:max_draw]):
        x1, y1 = kpts1[i1]
        x2, y2 = kpts2[i2]

        color = (0, 255, 0)
        if inliers_mask is not None and not inliers_mask[idx]:
            color = (0, 0, 255)

        p1 = (int(x1), int(y1))
        p2 = (int(x2) + w1, int(y2))

        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
        cv2.line(canvas, p1, p2, color, 1)

    return canvas


# OpenCV SIFT for Comparison 

def opencv_sift_detect_and_match(img1, img2, ratio=0.75):
    """
    Runs OpenCV's official SIFT implementation and matches features
    using BFMatcher + ratio test + RANSAC. Used to compare against
    our custom SIFT.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kps1, desc1 = sift.detectAndCompute(gray1, None)
    kps2, desc2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return kps1, kps2, [], None, None

    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    inliers = mask.ravel().astype(bool) if mask is not None else None

    return kps1, kps2, good_matches, H, inliers


def draw_opencv_matches(img1, img2, kps1, kps2, matches, inliers, max_draw=100):
    """
    Uses OpenCV’s built-in drawing function for match visualization.
    """
    matches_to_draw = matches[:max_draw]
    return cv2.drawMatches(
        img1, kps1,
        img2, kps2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


#  Main Script 

def main():
    if not os.path.exists(IMG1_PATH) or not os.path.exists(IMG2_PATH):
        print("[ERROR] Image paths are incorrect.")
        return

    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)

    if img1 is None or img2 is None:
        print("[ERROR] Failed to load images.")
        return

    # Run our SIFT 
    print("[INFO] Running custom SIFT...")
    sift_custom = SIFTFromScratch()
    kpts1_xy, descs1 = sift_custom.detect_and_compute(img1)
    kpts2_xy, descs2 = sift_custom.detect_and_compute(img2)

    print(f"[INFO] Custom SIFT: {len(kpts1_xy)} keypoints in img1, {len(kpts2_xy)} in img2")

    matches_custom = match_descriptors(descs1, descs2)
    print(f"[INFO] Custom SIFT: {len(matches_custom)} matches after ratio test")

    H_custom, inliers_custom = ransac_homography(kpts1_xy, kpts2_xy, matches_custom)
    num_inliers_custom = int(inliers_custom.sum()) if inliers_custom is not None else 0
    print(f"[INFO] Custom SIFT: {num_inliers_custom} inliers after RANSAC")

    canvas_custom = draw_matches(img1, img2, kpts1_xy, kpts2_xy,
                                 matches_custom, inliers_custom)
    cv2.imwrite(OUT_MATCHES_OURS, canvas_custom)
    print(f"[INFO] Saved custom SIFT matches: {OUT_MATCHES_OURS}")

    # ----- Run OpenCV SIFT -----
    print("[INFO] Running OpenCV SIFT...")
    kps1_cv, kps2_cv, matches_cv, H_cv, inliers_cv = opencv_sift_detect_and_match(img1, img2)

    print(f"[INFO] OpenCV SIFT: {len(kps1_cv)} keypoints in img1, {len(kps2_cv)} in img2")
    print(f"[INFO] OpenCV SIFT: {len(matches_cv)} matches after ratio test")

    if inliers_cv is not None:
        print(f"[INFO] OpenCV SIFT: {inliers_cv.sum()} inliers after RANSAC")
    else:
        print("[INFO] OpenCV SIFT: RANSAC failed or insufficient matches.")

    if matches_cv:
        canvas_cv = draw_opencv_matches(img1, img2, kps1_cv, kps2_cv,
                                        matches_cv, inliers_cv)
        cv2.imwrite(OUT_MATCHES_OPENCV, canvas_cv)
        print(f"[INFO] Saved OpenCV SIFT matches: {OUT_MATCHES_OPENCV}")


if __name__ == "__main__":
    main()
