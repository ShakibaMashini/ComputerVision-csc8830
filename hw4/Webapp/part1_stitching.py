import os
import cv2
from part1_code_original import stitch_sequence  # you paste your Part 1 full code in part1_code_original.py

def run_panorama_stitching(image_paths, output_folder):
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    print("[INFO] Running panorama stitching...")

    pano = stitch_sequence(images)
    output_path = os.path.join(output_folder, "panorama_result.png")

    cv2.imwrite(output_path, pano)
    print("[INFO] Panorama saved to:", output_path)

    return output_path
