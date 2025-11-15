import numpy as np
import matplotlib.pyplot as plt

# Camera intrinsic matrix from calibration (for a 1920x1080 image).
# Rows:
#   [fx,  0, cx]
#   [ 0, fy, cy]
#   [ 0,  0,  1]
# fx, fy: focal lengths in pixels
# cx, cy: principal point (image center in pixels)
K_CALIB = np.array([
    [1434.41,    0.00,  949.77],
    [   0.00, 1430.68,  541.41],
    [   0.00,    0.00,    1.00]
], dtype=float)

# Resolution used during camera calibration: (width, height)
CALIB_SIZE = (1920, 1080)

# Global state for mouse clicks
click_count = 0   # how many points have been clicked so far
uv1, uv2 = None, None  # first and second click locations in image coordinates


def rescale_K(K_calib: np.ndarray,
              calib_size: tuple[int, int],
              current_size: tuple[int, int]) -> np.ndarray:
    """
    Rescale the intrinsic matrix K when the image resolution is different
    from the calibration resolution.

    Arguments:
        K_calib     : intrinsic matrix for the calibration image size
        calib_size  : (W0, H0) width and height used during calibration
        current_size: (W, H) width and height of the current image

    Returns:
        K           : intrinsic matrix adjusted for the current image size
    """
    W0, H0 = calib_size
    W, H = current_size

    # Scale factors along x and y based on how the resolution changed
    sx, sy = W / W0, H / H0

    # Make a copy so we do not change the original matrix
    K = K_calib.copy()

    # Scale focal lengths and principal point according to new resolution
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy

    return K


def onclick(event):
    """
    Mouse click callback. Every time the user clicks on the image:
      - First click is stored as uv1
      - Second click is stored as uv2 and the figure is closed

    The event provides xdata, ydata in image coordinates.
    """
    global uv1, uv2, click_count

    # Ignore clicks outside the image area
    if event.xdata is None or event.ydata is None:
        return

    # Homogeneous pixel coordinates [u, v, 1]^T
    uv = np.array([[event.xdata],
                   [event.ydata],
                   [1.0]], dtype=float)

    if click_count == 0:
        # First point selected
        uv1 = uv
    else:
        # Second point selected
        uv2 = uv
        # Close the window so the script can continue
        plt.close()

    click_count += 1


def main():
    """
    Main function:
      1. Load the image.
      2. Adapt the intrinsic matrix K to the image resolution.
      3. Ask the user for the distance from the camera to the object (Zc).
      4. Let the user click two points on the object in the image.
      5. Back-project the pixels to 3D using Zc and K^{-1}.
      6. Compute and display the estimated real-world distance between the points.
    """
    global uv1, uv2, click_count

    # Path to the image you want to measure on
    img_path = "book.jpeg"
    image = plt.imread(img_path)
    H, W = image.shape[:2]

    # Rescale intrinsic matrix to match the current image resolution
    K = rescale_K(K_CALIB, CALIB_SIZE, (W, H))
    K_inv = np.linalg.inv(K)

    # Ask user about units and distance from camera to the object plane
    unit = (input("Units for Zc (e.g., cm, mm, m, in): ").strip() or "cm")
    zc = float(input(f"Distance from camera Zc (in {unit}): ").strip())

    # Reset click state before starting the interaction
    click_count = 0
    uv1 = uv2 = None

    # Show the image and wait for two clicks from the user
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Click TWO points on the object (inside the image)")
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    # If user closes the window without two valid clicks, abort
    if uv1 is None or uv2 is None:
        print("You must click two points inside the image. Try again.")
        return

    # Visualize the selected points and the line between them
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.scatter(uv1[0, 0], uv1[1, 0], color="pink", s=100, zorder=2)
    plt.scatter(uv2[0, 0], uv2[1, 0], color="pink", s=100, zorder=2)
    plt.plot([uv1[0, 0], uv2[0, 0]],
             [uv1[1, 0], uv2[1, 0]],
             color="pink",
             linewidth=2,
             zorder=1)

    # Distance between the two points in pixel space (just for reference)
    pix_dist = float(np.hypot(uv2[0, 0] - uv1[0, 0],
                              uv2[1, 0] - uv1[1, 0]))

    # Back-project the pixel coordinates into 3D assuming they lie on
    # a plane at depth Zc. For a pinhole camera model:
    #   [X, Y, Z]^T = Zc * K^{-1} * [u, v, 1]^T
    xyz1 = (K_inv @ uv1) * zc
    xyz2 = (K_inv @ uv2) * zc

    # Euclidean distance between the two 3D points
    length = float(np.linalg.norm(xyz1 - xyz2))

    # Show the estimated length on the image and save the figure
    title = f"estimated length: {length:.2f} {unit}   (pixel dist: {pix_dist:.1f})"
    plt.title(title)
    plt.savefig("measure_depth_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(title)
    print("Saved figure to measure_depth_result.png")


if __name__ == "__main__":
    main()
