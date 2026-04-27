#!/usr/bin/env python3
"""
Detect ArUco markers (IDs 10-13, DICT_5X5_100, 38 mm) in the live RealSense
feed and plot their poses in the robot base frame alongside the camera frame.

Prerequisites (all on the User PC):
  - roscore running
  - realsense2_camera driver running (roslaunch realsense2_camera rs_camera.launch)
  - calib_data/T_base_cam.npy present (hand-eye calibration already done)

Usage:
  python3 detect_markers_plot.py

Workflow:
  - Live RealSense window shows detected markers overlaid.
  - Press 's' in the camera window to snapshot and open a 3D matplotlib plot
    showing: robot base frame, camera frame, and each detected marker's frame.
  - Press 'q' to quit.

Output (per snapshot):
  - 3D matplotlib window
  - marker_poses.npz saved alongside the script with all T_base_marker matrices
"""
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

# ---- Config ----
CALIB_PATH = "spatial_calib/T_base_cam.npy"
MARKER_IDS = [10, 11, 12, 13]
MARKER_LEN = 0.038  # 38 mm
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# ---- ArUco detector (handles OpenCV 4.6 vs 4.7+) ----
try:
    _detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    def detect(img):
        return _detector.detectMarkers(img)
except AttributeError:
    _params = cv2.aruco.DetectorParameters_create()
    def detect(img):
        return cv2.aruco.detectMarkers(img, DICT, parameters=_params)

# Marker corners in marker-local frame (TL, TR, BR, BL) — matches detectMarkers output
h = MARKER_LEN / 2.0
OBJ_PTS = np.array(
    [[-h,  h, 0],
     [ h,  h, 0],
     [ h, -h, 0],
     [-h, -h, 0]], dtype=np.float32,
)

bridge = CvBridge()
latest_image = None
latest_K = None
latest_D = None


def image_cb(msg):
    global latest_image
    latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")


def camera_info_cb(msg):
    global latest_K, latest_D
    latest_K = np.array(msg.K).reshape(3, 3)
    latest_D = np.array(msg.D) if len(msg.D) else np.zeros(5)


def detect_and_pose(img, K, D):
    """Return dict {id: T_cam_marker (4x4)} for IDs in MARKER_IDS found in img."""
    results = {}
    corners, ids, _ = detect(img)
    if ids is None:
        return results
    ids_flat = ids.flatten()
    for i, mid in enumerate(ids_flat):
        if mid not in MARKER_IDS:
            continue
        img_pts = corners[i][0].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_PTS, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.flatten()
        results[int(mid)] = T
    return results


def overlay_markers(img, K, D, poses):
    """Draw detected marker outlines + axes on the image."""
    vis = img.copy()
    corners, ids, _ = detect(img)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    for mid, T in poses.items():
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(vis, K, D, rvec, tvec, MARKER_LEN * 0.8)
    return vis


def draw_frame(ax, T, name, scale=0.08, lw=3, alpha=1.0):
    o = T[:3, 3]
    colors = ["r", "g", "b"]  # X, Y, Z
    for i, c in enumerate(colors):
        end = o + scale * T[:3, i]
        ax.plot(*zip(o, end), color=c, linewidth=lw, alpha=alpha)
    ax.text(*o, f"  {name}", fontsize=10, fontweight="bold")


def set_equal_3d(ax, pts):
    pts = np.array(pts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    c = (mins + maxs) / 2.0
    r = max((maxs - mins).max() / 2.0 * 1.2, 0.1)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def plot_scene(T_base_cam, marker_poses_base):
    """3D plot of base + camera + all detected markers in base frame."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_frame(ax, np.eye(4), "base", scale=0.12, lw=4)
    draw_frame(ax, T_base_cam, "camera", scale=0.10, lw=3)
    ax.plot(
        *zip([0, 0, 0], T_base_cam[:3, 3]),
        color="k", linestyle="--", alpha=0.35, linewidth=1,
    )

    pts_for_scaling = [np.zeros(3), T_base_cam[:3, 3]]
    for mid, T_bm in sorted(marker_poses_base.items()):
        draw_frame(ax, T_bm, f"ID{mid}", scale=0.05, lw=2.5)
        pts_for_scaling.append(T_bm[:3, 3])

    if marker_poses_base:
        ax.scatter(
            *np.array([T[:3, 3] for T in marker_poses_base.values()]).T,
            color="orange", s=60, alpha=0.9, edgecolors="k",
        )

    set_equal_3d(ax, pts_for_scaling)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Markers in robot base frame  ({len(marker_poses_base)} detected)"
    )
    plt.tight_layout()
    plt.show()


def main():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(
            f"{CALIB_PATH} not found. Run compute_calibration.py first."
        )
    T_base_cam = np.load(CALIB_PATH)
    print(f"Loaded T_base_cam from {CALIB_PATH}")
    print(f"  Camera origin in base: {T_base_cam[:3, 3].round(3)} m\n")

    rospy.init_node("marker_detector", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_cb)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, camera_info_cb)

    print("Waiting for camera topics...")
    while not rospy.is_shutdown() and (latest_image is None or latest_K is None):
        rospy.sleep(0.05)
    print("Camera feed live. Press 's' to snapshot & plot, 'q' to quit.\n")

    while not rospy.is_shutdown():
        if latest_image is None:
            rospy.sleep(0.03); continue
        img = latest_image.copy()
        poses_cam = detect_and_pose(img, latest_K, latest_D)
        vis = overlay_markers(img, latest_K, latest_D, poses_cam)

        # Live readout in the image
        y = 25
        cv2.putText(vis, f"Detected IDs: {sorted(poses_cam.keys())}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        for mid, T in sorted(poses_cam.items()):
            t = T[:3, 3]
            cv2.putText(vis, f"ID{mid} cam: x={t[0]:+.2f} y={t[1]:+.2f} z={t[2]:+.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 20
        cv2.putText(vis, "[s] snapshot + 3D plot  [q] quit",
                    (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("detector", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            # Transform each detected marker into base frame
            marker_poses_base = {
                mid: T_base_cam @ T_cam for mid, T_cam in poses_cam.items()
            }

            print(f"\n--- Snapshot: {len(marker_poses_base)} markers in base frame ---")
            for mid, T in sorted(marker_poses_base.items()):
                t = T[:3, 3]
                print(f"  ID{mid}:  x={t[0]:+.3f}  y={t[1]:+.3f}  z={t[2]:+.3f}  m")

            # Save to disk
            np.savez(
                "marker_poses.npz",
                T_base_cam=T_base_cam,
                **{f"ID{mid}": T for mid, T in marker_poses_base.items()},
            )
            print("Saved poses to marker_poses.npz")

            plot_scene(T_base_cam, marker_poses_base)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
