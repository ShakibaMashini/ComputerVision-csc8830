# depth_utils.py

import numpy as np
from stereo_config import fx, fy, cx, cy, BASELINE

def disparity_to_depth(disparity: float,
                       f: float = fx,
                       baseline: float = BASELINE) -> float:
    """
    Convert disparity (in pixels) to depth Z (same units as baseline).

    Parameters
    ----------
    disparity : float
        x_left - x_right in pixels. Must be > 0.
    f : float
        Focal length in pixels.
    baseline : float
        Baseline between the two cameras.

    Returns
    -------
    float
        Depth Z in same units as baseline (e.g. meters).
    """
    if disparity <= 0:
        # Very small or negative disparity is invalid / too far
        return np.inf
    return (f * baseline) / disparity


def image_to_camera_coords(u: float, v: float, Z: float,
                           f_x: float = fx, f_y: float = fy,
                           c_x: float = cx, c_y: float = cy):
    """
    Convert image coordinates (u, v) + depth Z into camera coordinates (X, Y, Z).

    Parameters
    ----------
    u, v : float
        Pixel coordinates in the rectified left image.
    Z : float
        Depth value from disparity_to_depth.

    Returns
    -------
    (X, Y, Z) : tuple of floats
        3D coordinates in camera coordinate frame.
    """
    X = (u - c_x) * Z / f_x
    Y = (v - c_y) * Z / f_y
    return X, Y, Z


def distance_3d(p1, p2):
    """
    Euclidean distance between two 3D points p1 and p2.
    Each is (X, Y, Z).
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))
