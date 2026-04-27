#!/usr/bin/env python3
"""
Visualize the hand-eye calibration: robot base frame + camera frame in 3D.

Run on the User PC (or your laptop) with numpy + matplotlib:
    python3 plot_calibration.py

Optionally also shows:
  - All recorded EE positions as grey dots (context for the calibration poses)
  - All detected marker positions in base frame as small coloured dots
    (computed via T_base_marker = T_base_cam @ T_cam_marker).
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "calib_data"
MARKER_LEN = 0.04
MARKER_ID = 0

T_base_cam = np.load(f"{DATA_DIR}/T_base_cam.npy")


def draw_frame(ax, T, name, scale=0.1, lw=3, alpha=1.0):
    """Draw an RGB (x=red, y=green, z=blue) triad at the given 4x4 pose."""
    origin = T[:3, 3]
    x_axis = origin + scale * T[:3, 0]
    y_axis = origin + scale * T[:3, 1]
    z_axis = origin + scale * T[:3, 2]

    ax.plot(*zip(origin, x_axis), color="r", linewidth=lw, alpha=alpha)
    ax.plot(*zip(origin, y_axis), color="g", linewidth=lw, alpha=alpha)
    ax.plot(*zip(origin, z_axis), color="b", linewidth=lw, alpha=alpha)
    ax.text(*origin, f"  {name}", fontsize=10, fontweight="bold")


def set_equal_3d(ax, pts):
    """Force equal aspect ratio on a 3D axes, given a Nx3 point array."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 * 1.2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def try_load_context():
    """Load recorded EE translations and recompute marker positions in base frame."""
    ee_pts = []
    marker_pts = []

    # Intrinsics for PnP
    try:
        K = np.load(f"{DATA_DIR}/K.npy")
        D = np.load(f"{DATA_DIR}/D.npy")
    except FileNotFoundError:
        K = D = None

    DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    try:
        detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
        detect = lambda img: detector.detectMarkers(img)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        detect = lambda img: cv2.aruco.detectMarkers(img, DICT, parameters=params)

    h = MARKER_LEN / 2.0
    obj_pts = np.array(
        [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32
    )

    for img_path in sorted(glob.glob(f"{DATA_DIR}/images/*.png")):
        idx = os.path.basename(img_path).split(".")[0]
        T_path = f"{DATA_DIR}/images/{idx}_T_base_ee.npy"
        if not os.path.exists(T_path):
            continue

        T_base_ee = np.load(T_path)
        ee_pts.append(T_base_ee[:3, 3])

        if K is None:
            continue
        img = cv2.imread(img_path)
        corners, ids, _ = detect(img)
        if ids is None or MARKER_ID not in ids.flatten():
            continue
        i = np.where(ids.flatten() == MARKER_ID)[0][0]
        img_pts = corners[i][0].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3], _ = cv2.Rodrigues(rvec)
        T_cam_marker[:3, 3] = tvec.flatten()
        T_base_marker = T_base_cam @ T_cam_marker
        marker_pts.append(T_base_marker[:3, 3])

    ee_pts = np.array(ee_pts) if ee_pts else np.empty((0, 3))
    marker_pts = np.array(marker_pts) if marker_pts else np.empty((0, 3))
    return ee_pts, marker_pts


def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Robot base frame at origin
    T_base = np.eye(4)
    draw_frame(ax, T_base, "robot base", scale=0.12)

    # Camera frame at T_base_cam
    draw_frame(ax, T_base_cam, "camera", scale=0.10)

    # Line from base to camera to visualize offset
    ax.plot(*zip(T_base[:3, 3], T_base_cam[:3, 3]),
            color="k", linestyle="--", alpha=0.4, linewidth=1)

    ee_pts, marker_pts = try_load_context()
    if len(ee_pts):
        ax.scatter(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2],
                   color="grey", s=20, alpha=0.5, label="EE positions")
    if len(marker_pts):
        ax.scatter(marker_pts[:, 0], marker_pts[:, 1], marker_pts[:, 2],
                   color="orange", s=30, alpha=0.7,
                   label="marker in base (via calibration)")

    # Aspect
    all_pts = np.vstack([[0, 0, 0], T_base_cam[:3, 3], ee_pts, marker_pts]) \
        if len(ee_pts) or len(marker_pts) else np.vstack([[0, 0, 0], T_base_cam[:3, 3]])
    set_equal_3d(ax, all_pts)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Hand-eye calibration: robot base vs. camera frame")
    if len(ee_pts) or len(marker_pts):
        ax.legend(loc="upper right")

    # Print numeric summary
    t = T_base_cam[:3, 3]
    R = T_base_cam[:3, :3]
    euler_deg = np.degrees(
        np.array([
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)),
            np.arctan2(R[1, 0], R[0, 0]),
        ])
    )
    print(f"Camera origin (base frame):  {t.round(3)} m")
    print(f"Camera orientation (XYZ euler, deg): {euler_deg.round(1)}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
