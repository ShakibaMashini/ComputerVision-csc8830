import argparse
import cv2
import numpy as np


# CAMERA SETUP

def open_camera(device_index=0):
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access Mac camera. Check permissions.")
    return cap


# CREATE TRACKER

def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise RuntimeError("CSRT tracker not available in this OpenCV installation.")


# -------------------------------------------------------------
# PART (i) — ARUCO MARKER TRACKER
# -------------------------------------------------------------
def run_aruco_tracker():
    print("Running ArUco tracker (Press ESC to exit).")

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco missing! Install contrib package:\n"
            "pip install opencv-contrib-python"
        )

    aruco = cv2.aruco

    # Dictionary (compatibility)
    if hasattr(aruco, "getPredefinedDictionary"):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    else:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)

    # Parameters (compatibility)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # Detector (new API)
    use_new = hasattr(aruco, "ArucoDetector")
    if use_new:
        detector = aruco.ArucoDetector(dictionary, parameters)

    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers (new and old API)
        if use_new:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            corners, ids, rejected = aruco.detectMarkers(
                frame, dictionary, parameters=parameters
            )

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            pts = np.concatenate(corners, axis=1)
            x_min = int(pts[:, :, 0].min())
            x_max = int(pts[:, :, 0].max())
            y_min = int(pts[:, :, 1].min())
            y_max = int(pts[:, :, 1].max())

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "ArUco Tracking",
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Marker Detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("ArUco Marker Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------
# PART (ii) — MARKERLESS TRACKING
# -------------------------------------------------------------
def run_markerless_tracker():
    print("Markerless Tracking")
    print("Press 's' to select an ROI. Press ESC to quit.")

    cap = open_camera()
    tracker = None
    initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if initialized:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Tracking Lost",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to select object",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

        cv2.imshow("Markerless Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not initialized:
            roi = cv2.selectROI("Markerless Tracker", frame, False, False)
            if roi[2] > 0 and roi[3] > 0:
                tracker = create_tracker()
                tracker.init(frame, roi)
                initialized = True
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# def run_sam2_npz_tracker(npz_path):
#     print("SAM2 NPZ Tracker")
#     print(f"Using NPZ: {npz_path}")

#     data = np.load(npz_path)
#     if "init_bbox" not in data:
#         raise ValueError("NPZ must contain 'init_bbox' array.")

#     # Convert to native Python floats
#     bbox = data["init_bbox"].astype(float)
#     bbox = tuple(map(float, bbox))

#     print("Loaded bbox:", bbox)

#     cap = open_camera()
#     tracker = create_tracker()
#     initialized = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if not initialized:
#             tracker.init(frame, bbox)
#             initialized = True

#         success, box = tracker.update(frame)
#         if success:
#             x, y, w, h = map(int, box)
#             cv2.rectangle(frame, (x, y), (x + w, y + h),
#                           (255, 0, 0), 3)
#             cv2.putText(frame, "SAM2 Tracking",
#                         (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (255, 0, 0), 2)
#         else:
#             cv2.putText(frame, "Tracking Lost",
#                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 0, 255), 2)

#         cv2.imshow("SAM2 Tracker", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()



# -------------------------------------------------------------
# PART (iii) — SAM2 NPZ-BASED TRACKING
# -------------------------------------------------------------
# def run_sam2_npz_tracker(npz_path):
#     print("SAM2 NPZ Tracker")
#     print(f"Using NPZ: {npz_path}")

#     data = np.load(npz_path)
#     if "init_bbox" not in data:
#         raise ValueError("NPZ must contain 'init_bbox' array.")

#     # Convert to Python floats
#     bbox = data["init_bbox"].astype(float)
#     bbox = tuple(map(float, bbox))

#     print("Loaded bbox:", bbox)

#     cap = open_camera()
#     tracker = create_tracker()
#     initialized = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Initialize tracker once, AFTER getting frame size
#         if not initialized:
#             H, W = frame.shape[:2]

#             x, y, w, h = bbox

#             # ---------- OPTION 1: CLAMP BBOX TO FRAME SIZE ----------
#             w = min(w, W - x - 1)
#             h = min(h, H - y - 1)
#             bbox = (float(x), float(y), float(w), float(h))
#             print("Clamped bbox:", bbox)
#             # --------------------------------------------------------

#             tracker.init(frame, bbox)
#             initialized = True

#         success, box = tracker.update(frame)
#         if success:
#             x, y, w, h = map(int, box)
#             cv2.rectangle(frame, (x, y), (x + w, y + h),
#                           (255, 0, 0), 3)
#             cv2.putText(frame, "SAM2 Tracking",
#                         (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (255, 0, 0), 2)
#         else:
#             cv2.putText(frame, "Tracking Lost",
#                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 0, 255), 2)

#         cv2.imshow("SAM2 Tracker", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def run_sam2_npz_tracker(npz_path):
    print("SAM2 NPZ Tracker")
    print(f"Using NPZ: {npz_path}")

    # data = np.load(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "init_bbox" not in data:
        raise ValueError("NPZ must contain 'init_bbox' array.")

    # ---- STRONGEST POSSIBLE CONVERSION -----
    raw_bbox = data["init_bbox"]      # numpy array
    raw_bbox = raw_bbox.tolist()      # python list (but may contain numpy floats)
    bbox = tuple(float(v) for v in raw_bbox)   # each element pure Python float
    # -----------------------------------------

    print("Loaded bbox:", bbox)
    print("Types:", [type(v) for v in bbox])  # MUST all be <class 'float'>

    cap = open_camera()
    tracker = create_tracker()
    initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not initialized:
            H, W = frame.shape[:2]

            x, y, w, h = bbox

            # ---- CLAMP TO IMAGE SIZE ----
            w = min(w, W - x - 1)
            h = min(h, H - y - 1)
            # bbox = (float(x), float(y), float(w), float(h))
            bbox = (int(x), int(y), int(w), int(h))
         

            print("Clamped bbox:", bbox)
            print("Clamped Types:", [type(v) for v in bbox])

            tracker.init(frame, bbox)
            initialized = True

        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (255, 0, 0), 3)
            cv2.putText(frame, "SAM2 Tracking",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking Lost",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("SAM2 Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["aruco", "markerless", "sam2"])
    parser.add_argument("--npz", type=str, default="sam2_output.npz")

    args = parser.parse_args()

    if args.mode == "aruco":
        run_aruco_tracker()
    elif args.mode == "markerless":
        run_markerless_tracker()
    else:
        run_sam2_npz_tracker(args.npz)


if __name__ == "__main__":
    main()





