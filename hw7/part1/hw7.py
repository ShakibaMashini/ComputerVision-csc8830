import cv2
import numpy as np

from depth_utils import (
    disparity_to_depth,
    image_to_camera_coords,
    distance_3d,
)
from stereo_config import fx, fy, cx, cy, BASELINE


#CONFIG
LEFT_IMG_PATH  = "left.jpg"
RIGHT_IMG_PATH = "right.jpg"


#  GLOBALS FOR MOUSE CALLBACK
left_img = None
disp_map = None
clicked_points_img = []      # list of (u, v) in image coords
clicked_points_3d = []       # list of (X, Y, Z) in camera coords


def compute_disparity(left_img, right_img):
    """
    Compute disparity map using StereoSGBM from rectified stereo pair.
    """
    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 128   # must be divisible by 16
    block_size = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disp


def on_mouse_left(event, x, y, flags, param):
    """
    Mouse callback for left image window.
    We collect clicks, convert to 3D, and measure distance.
    """
    global clicked_points_img, clicked_points_3d

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Read disparity at this pixel
    d = disp_map[y, x]

    # Compute depth Z
    Z = disparity_to_depth(d)
    if not np.isfinite(Z):
        print(f"Click at ({x}, {y}) has invalid disparity ({d:.3f}).")
        return

    # Convert to camera coordinates
    X, Y, Z = image_to_camera_coords(x, y, Z)
    clicked_points_img.append((x, y))
    clicked_points_3d.append((X, Y, Z))

    print(f"Click #{len(clicked_points_img)} at pixel ({x}, {y}) -> 3D ({X:.3f}, {Y:.3f}, {Z:.3f})")

    # Draw a small circle where we clicked
    cv2.circle(left_img, (x, y), 5, (0, 0, 255), -1)

    # If we have 2 points, compute and show distance
    if len(clicked_points_img) == 2:
        p1_3d, p2_3d = clicked_points_3d
        dist = distance_3d(p1_3d, p2_3d)

        print(f"3D distance between points: {dist:.3f} (same units as BASELINE)")

        # Draw a line between the two points
        cv2.line(left_img, clicked_points_img[0], clicked_points_img[1], (0, 255, 0), 2)

        # Put text on image
        mid_x = (clicked_points_img[0][0] + clicked_points_img[1][0]) // 2
        mid_y = (clicked_points_img[0][1] + clicked_points_img[1][1]) // 2
        cv2.putText(
            left_img,
            f"{dist:.2f}",
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Reset so you can measure another segment
        clicked_points_img = []
        clicked_points_3d = []


def main():
    global left_img, disp_map

    # LOAD IMAGES
    left_img = cv2.imread(LEFT_IMG_PATH)
    right_img = cv2.imread(RIGHT_IMG_PATH)

    if left_img is None or right_img is None:
        raise FileNotFoundError("Could not load stereo images. Check LEFT_IMG_PATH and RIGHT_IMG_PATH.")

    # COMPUTE DISPARITY 
    print("Computing disparity map...")
    disp_map = compute_disparity(left_img, right_img)
    print("Disparity map computed.")

    # Normalize disparity for visualization only
    disp_vis = disp_map.copy()
    disp_vis[~np.isfinite(disp_vis)] = 0
    disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)

    # SETUP WINDOWS 
    cv2.namedWindow("Left")
    cv2.namedWindow("Right")
    cv2.namedWindow("Disparity")

    cv2.setMouseCallback("Left", on_mouse_left)

    while True:
        cv2.imshow("Left", left_img)
        cv2.imshow("Right", right_img)
        cv2.imshow("Disparity", disp_vis)

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):   # ESC or q to quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
