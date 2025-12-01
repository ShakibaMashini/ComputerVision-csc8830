import os
import cv2
from part2_code_original import (
    SIFTFromScratch, match_descriptors, ransac_homography,
    draw_matches, opencv_sift_detect_and_match, draw_opencv_matches
)

def run_sift_comparison(img1_path, img2_path, output_folder):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("[INFO] Running custom SIFT...")
    sift_custom = SIFTFromScratch()
    k1, d1 = sift_custom.detect_and_compute(img1)
    k2, d2 = sift_custom.detect_and_compute(img2)

    matches = match_descriptors(d1, d2, ratio=0.75)
    H, inliers = ransac_homography(k1, k2, matches, num_iter=2000)

    out_custom = os.path.join(output_folder, "sift_custom.png")
    canvas = draw_matches(img1, img2, k1, k2, matches, inliers)
    cv2.imwrite(out_custom, canvas)

    print("[INFO] Running OpenCV SIFT...")
    kps1_cv, kps2_cv, matches_cv, H_cv, inliers_cv = opencv_sift_detect_and_match(img1, img2)

    out_cv = os.path.join(output_folder, "sift_opencv.png")
    canvas2 = draw_opencv_matches(img1, img2, kps1_cv, kps2_cv, matches_cv, inliers_cv)
    cv2.imwrite(out_cv, canvas2)

    return out_custom, out_cv
