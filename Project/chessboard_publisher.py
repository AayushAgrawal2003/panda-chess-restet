#!/usr/bin/env python3
"""
Detect the half-chessboard (ArUco corners 10/11/12/13) in the live RealSense
feed and publish a PoseArray of all 32 square poses in the FRANKA BASE frame.

Each pose:
  - position    = centre of the chess square on the board surface
  - orientation = rotation whose +Z axis equals the board surface normal,
                  FORCED to point UP in the base frame (+Z_base)

Published topics:
  /chessboard/poses     -- geometry_msgs/PoseArray (32 poses)
  /chessboard/viz       -- visualization_msgs/MarkerArray (32 arrows for RViz)
  /chessboard/square_XX -- geometry_msgs/PoseStamped per square (XX = 00..31)

Prerequisites (all on the User PC):
  - roscore running
  - realsense2_camera driver running
  - spatial_calib/T_base_cam.npy present (hand-eye calibration done)

Usage:
    python3 chessboard_publisher.py
Keyboard (in the preview window):
    q -- quit

Geometry / indexing (paper frame):
  - Origin = top-left CORNER of chessboard
  - +X right (toward col 7), +Y down (toward row 3), +Z out of page
  - square index = r*8 + c  (r in 0..3 top->bottom, c in 0..7 left->right)
"""
import os
import numpy as np
import cv2
import rospy
import tf.transformations as tft
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

# ---------------- Config ----------------
CALIB_PATH = "spatial_calib/T_base_cam.npy"
BASE_FRAME = "panda_link0"

MARKER_SIZE = 0.038
MARGIN      = 0.006
SQUARE_SIZE = 0.050
ROWS, COLS  = 4, 8
M = MARKER_SIZE / 2 + MARGIN  # marker centre offset from nearest chessboard corner

MARKER_IDS = (10, 11, 12, 13)
DICT_FLAG = cv2.aruco.DICT_5X5_100

# Marker-centre positions in paper frame (TL, TR, BL, BR)
MARKER_CENTERS_PAPER = {
    10: np.array([-M,                    -M,                    0.0]),
    11: np.array([COLS * SQUARE_SIZE + M, -M,                    0.0]),
    12: np.array([-M,                    ROWS * SQUARE_SIZE + M, 0.0]),
    13: np.array([COLS * SQUARE_SIZE + M, ROWS * SQUARE_SIZE + M, 0.0]),
}

# 32 square centres in paper frame
SQUARE_CENTERS_PAPER = np.array([
    [(c + 0.5) * SQUARE_SIZE, (r + 0.5) * SQUARE_SIZE, 0.0]
    for r in range(ROWS) for c in range(COLS)
], dtype=np.float32)

# ---------------- ArUco detector ----------------
DICT = cv2.aruco.getPredefinedDictionary(DICT_FLAG)
try:
    _det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    def detect(img): return _det.detectMarkers(img)
except AttributeError:
    _p = cv2.aruco.DetectorParameters_create()
    def detect(img): return cv2.aruco.detectMarkers(img, DICT, parameters=_p)

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


# ---------------- Orientation helpers ----------------
def rotation_z_up(R_board_base):
    """
    Given a 3x3 rotation that represents the board's frame in base,
    return a rotation whose +Z axis is the UP direction in base frame
    (i.e., +Z_base = [0,0,1]) while preserving the board's in-plane +X
    direction projected onto the horizontal plane.

    If the board is flat on the table and camera looks down, the detected
    normal could point either up or down depending on PnP's sign choice.
    This function normalises to always point up.
    """
    x_board = R_board_base[:, 0]
    n_board = R_board_base[:, 2]

    # If detected normal points down in base, flip by rotating 180 deg around
    # the board's +X axis (keeps +X, flips +Y and +Z).
    if n_board[2] < 0:
        flip = np.diag([1.0, -1.0, -1.0])
        R_board_base = R_board_base @ flip
        x_board = R_board_base[:, 0]
        n_board = R_board_base[:, 2]

    # Now force Z to be exactly UP in base, keep in-plane X as close as possible
    z_new = np.array([0.0, 0.0, 1.0])
    x_new = x_board - np.dot(x_board, z_new) * z_new
    nx = np.linalg.norm(x_new)
    if nx < 1e-6:  # degenerate: board X was almost vertical
        x_new = np.array([1.0, 0.0, 0.0])
    else:
        x_new /= nx
    y_new = np.cross(z_new, x_new)
    return np.column_stack([x_new, y_new, z_new])


def matrix_to_quat(R):
    T = np.eye(4); T[:3, :3] = R
    return tft.quaternion_from_matrix(T)  # (x, y, z, w)


def build_pose(pos, R):
    q = matrix_to_quat(R)
    p = Pose()
    p.position.x = float(pos[0])
    p.position.y = float(pos[1])
    p.position.z = float(pos[2])
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


def build_marker(i, pos, R, stamp, frame, life=0.5):
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = "chessboard_normals"
    m.id = i
    m.type = Marker.ARROW
    m.action = Marker.ADD
    start = Point(*pos)
    end = Point(*(pos + R[:, 2] * 0.04))  # 4 cm arrow along +Z of pose
    m.points = [start, end]
    m.scale.x = 0.004  # shaft diameter
    m.scale.y = 0.008  # head diameter
    m.scale.z = 0.008  # head length
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 1.0
    m.lifetime = rospy.Duration(life)
    return m


# ---------------- Main ----------------
def main():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(f"{CALIB_PATH} not found")
    T_base_cam = np.load(CALIB_PATH)
    rospy.loginfo(f"T_base_cam loaded. Camera origin in base: "
                  f"{T_base_cam[:3, 3].round(3)} m")

    rospy.init_node("chessboard_publisher", anonymous=False)
    rospy.Subscriber("/camera/color/image_raw", Image, image_cb)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, camera_info_cb)

    pub_arr = rospy.Publisher("/chessboard/poses", PoseArray, queue_size=1)
    pub_viz = rospy.Publisher("/chessboard/viz", MarkerArray, queue_size=1)
    pubs_individual = [
        rospy.Publisher(f"/chessboard/square_{i:02d}", PoseStamped, queue_size=1)
        for i in range(ROWS * COLS)
    ]

    rospy.loginfo("Waiting for camera...")
    rate = rospy.Rate(30)
    while not rospy.is_shutdown() and (latest_image is None or latest_K is None):
        rate.sleep()
    rospy.loginfo(f"Publishing on /chessboard/poses in frame '{BASE_FRAME}'")

    obj_pts = np.stack([MARKER_CENTERS_PAPER[i] for i in MARKER_IDS]
                       ).astype(np.float32)

    while not rospy.is_shutdown():
        if latest_image is None:
            rate.sleep(); continue
        img = latest_image.copy()
        corners, ids, _ = detect(img)

        vis = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        n_found = 0 if ids is None else sum(int(i) in MARKER_IDS
                                            for i in ids.flatten())
        status_txt = f"corners: {n_found}/4"

        if ids is not None and n_found == 4:
            ids_flat = ids.flatten()
            centers_px = {int(ids_flat[i]): corners[i][0].mean(axis=0)
                          for i in range(len(ids_flat)) if int(ids_flat[i]) in MARKER_IDS}
            img_pts = np.stack([centers_px[i] for i in MARKER_IDS]
                               ).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, latest_K, latest_D,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                # T_cam_paper
                R_cp, _ = cv2.Rodrigues(rvec)
                T_cam_paper = np.eye(4)
                T_cam_paper[:3, :3] = R_cp
                T_cam_paper[:3, 3] = tvec.flatten()

                # T_base_paper
                T_base_paper = T_base_cam @ T_cam_paper

                # Orientation forced so that +Z is UP in base
                R_paper_base = T_base_paper[:3, :3]
                R_square = rotation_z_up(R_paper_base)

                # Transform all 32 square centres to base frame
                squares_paper_h = np.hstack([SQUARE_CENTERS_PAPER,
                                             np.ones((ROWS * COLS, 1))])
                squares_base = (T_base_paper @ squares_paper_h.T).T[:, :3]

                stamp = rospy.Time.now()

                # PoseArray
                pa = PoseArray()
                pa.header.stamp = stamp
                pa.header.frame_id = BASE_FRAME
                pa.poses = [build_pose(squares_base[i], R_square)
                            for i in range(ROWS * COLS)]
                pub_arr.publish(pa)

                # Individual PoseStamped
                for i in range(ROWS * COLS):
                    ps = PoseStamped()
                    ps.header.stamp = stamp
                    ps.header.frame_id = BASE_FRAME
                    ps.pose = pa.poses[i]
                    pubs_individual[i].publish(ps)

                # RViz markers
                ma = MarkerArray()
                ma.markers = [build_marker(i, squares_base[i], R_square,
                                           stamp, BASE_FRAME)
                              for i in range(ROWS * COLS)]
                pub_viz.publish(ma)

                # Overlay: project square centres back to image for visual feedback
                proj, _ = cv2.projectPoints(SQUARE_CENTERS_PAPER, rvec, tvec,
                                            latest_K, latest_D)
                for i, p in enumerate(proj.reshape(-1, 2)):
                    pt = tuple(p.astype(int))
                    cv2.circle(vis, pt, 6, (0, 255, 255), -1)
                    cv2.putText(vis, f"{i}", (pt[0] - 12, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(vis, f"{i}", (pt[0] - 12, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                status_txt += (f"   published 32 poses   normal_in_base="
                               f"[{R_square[0, 2]:+.2f},"
                               f"{R_square[1, 2]:+.2f},"
                               f"{R_square[2, 2]:+.2f}]")

        cv2.putText(vis, status_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "q = quit", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("chessboard_publisher", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
