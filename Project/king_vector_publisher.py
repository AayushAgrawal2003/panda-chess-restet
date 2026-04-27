#!/usr/bin/env python3
"""
Detect the king's ArUco marker (DICT_4X4_50, 2.5 cm) and publish a
PoseStamped that encodes:
  - position    = cube center (1.5 cm behind marker plane along marker -Z)
  - orientation = rotation whose +Z axis aligns with the king-vector
                  (direction from cube center -> king's head, which is
                   the marker's +Y axis rotated into the robot base frame)

Topic : /king_vector   (ROS doesn't allow hyphens in names; using underscore)
Node  : king_vector_publisher
Frame : panda_link0  (change BASE_FRAME if your TF uses a different base)

Prerequisites (all on the User PC):
  - roscore running
  - realsense2_camera driver running
  - spatial_calib/T_base_cam.npy present

Usage:
    python3 king_vector_publisher.py

Keyboard (in the detector preview window):
    q - quit
"""
import os
import numpy as np
import cv2
import rospy
import tf.transformations as tft
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

# ---------- Config ----------
CALIB_PATH = "spatial_calib/T_base_cam.npy"
TARGET_ID = 2                 # the king's ArUco marker ID
MARKER_LEN = 0.025            # 2.5 cm
CUBE_DEPTH_BEHIND_MARKER = 0.015  # 1.5 cm: cube center offset along marker -Z
DICT_FLAG = cv2.aruco.DICT_4X4_50
BASE_FRAME = "panda_link0"    # robot base TF frame
TOPIC_POSE = "/king_vector"
TOPIC_MARKER_VIZ = "/king_vector/viz"  # optional RViz arrow marker

# ---------- ArUco setup ----------
DICT = cv2.aruco.getPredefinedDictionary(DICT_FLAG)
try:
    _det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    def detect(img): return _det.detectMarkers(img)
except AttributeError:
    _p = cv2.aruco.DetectorParameters_create()
    def detect(img): return cv2.aruco.detectMarkers(img, DICT, parameters=_p)

# Marker corners in marker-local frame (TL, TR, BR, BL) -- matches detector output
h = MARKER_LEN / 2.0
OBJ_PTS = np.array(
    [[-h,  h, 0], [ h,  h, 0], [ h, -h, 0], [-h, -h, 0]], dtype=np.float32,
)

# Cube center in marker frame (homogeneous)
P_MARKER_TO_CUBE_H = np.array([0.0, 0.0, -CUBE_DEPTH_BEHIND_MARKER, 1.0])

# King direction in marker frame (+Y is "up" along the piece toward head)
KING_DIR_MARKER = np.array([0.0, 1.0, 0.0])


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


def align_z_to_direction(v):
    """Build a 3x3 rotation matrix whose +Z column equals unit(v).
    X and Y are chosen via a stable cross-product trick."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, v); x /= np.linalg.norm(x)
    y = np.cross(v, x)
    R = np.column_stack([x, y, v])
    return R


def build_posestamped(cube_center_base, king_dir_base, frame, stamp):
    """Build PoseStamped with position = cube center, +Z aligned with king_dir."""
    R = align_z_to_direction(king_dir_base)
    T = np.eye(4); T[:3, :3] = R
    qx, qy, qz, qw = tft.quaternion_from_matrix(T)

    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.pose.position.x = float(cube_center_base[0])
    msg.pose.position.y = float(cube_center_base[1])
    msg.pose.position.z = float(cube_center_base[2])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def build_rviz_arrow(cube_center_base, king_dir_base, frame, stamp, length=0.08):
    """RViz arrow from cube center along the king direction for sanity visualisation."""
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = "king_vector"
    m.id = 0
    m.type = Marker.ARROW
    m.action = Marker.ADD
    from geometry_msgs.msg import Point
    start = Point(*cube_center_base)
    end = Point(*(cube_center_base + king_dir_base / np.linalg.norm(king_dir_base) * length))
    m.points = [start, end]
    m.scale.x = 0.006  # shaft diameter
    m.scale.y = 0.012  # head diameter
    m.scale.z = 0.012  # head length
    m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.0, 0.9, 1.0  # purple
    m.lifetime = rospy.Duration(0.5)
    return m


def main():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(f"{CALIB_PATH} not found. Run spatial calibration first.")
    T_base_cam = np.load(CALIB_PATH)
    rospy.loginfo(f"Loaded T_base_cam; camera origin in base = {T_base_cam[:3, 3].round(3)}")

    rospy.init_node("king_vector_publisher", anonymous=False)
    rospy.Subscriber("/camera/color/image_raw", Image, image_cb)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, camera_info_cb)

    pub_pose = rospy.Publisher(TOPIC_POSE, PoseStamped, queue_size=1)
    pub_viz  = rospy.Publisher(TOPIC_MARKER_VIZ, Marker, queue_size=1)

    rospy.loginfo("Waiting for camera feed + camera_info...")
    rate = rospy.Rate(30)
    while not rospy.is_shutdown() and (latest_image is None or latest_K is None):
        rate.sleep()
    rospy.loginfo(f"Publishing on {TOPIC_POSE}  (frame_id = {BASE_FRAME})")
    rospy.loginfo("Press 'q' in the preview window to quit.")

    while not rospy.is_shutdown():
        if latest_image is None:
            rate.sleep(); continue
        img = latest_image.copy()
        corners, ids, _ = detect(img)

        vis = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        target_found = False
        if ids is not None and TARGET_ID in ids.flatten():
            i = int(np.where(ids.flatten() == TARGET_ID)[0][0])
            img_pts = corners[i][0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                OBJ_PTS, img_pts, latest_K, latest_D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3], _ = cv2.Rodrigues(rvec)
                T_cam_marker[:3, 3] = tvec.flatten()

                # Base-frame transforms
                T_base_marker = T_base_cam @ T_cam_marker
                cube_center_base = (T_base_marker @ P_MARKER_TO_CUBE_H)[:3]
                king_dir_base = T_base_marker[:3, :3] @ KING_DIR_MARKER

                stamp = rospy.Time.now()
                pose_msg = build_posestamped(cube_center_base, king_dir_base,
                                             BASE_FRAME, stamp)
                pub_pose.publish(pose_msg)
                pub_viz.publish(build_rviz_arrow(cube_center_base, king_dir_base,
                                                 BASE_FRAME, stamp))
                target_found = True

                # On-screen overlays
                cv2.drawFrameAxes(vis, latest_K, latest_D, rvec, tvec,
                                  MARKER_LEN * 0.8, 2)
                kd = king_dir_base / np.linalg.norm(king_dir_base)
                cv2.putText(vis,
                            f"ID{TARGET_ID} cube@base = "
                            f"({cube_center_base[0]:+.3f}, "
                            f"{cube_center_base[1]:+.3f}, "
                            f"{cube_center_base[2]:+.3f}) m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(vis,
                            f"king_dir_base = "
                            f"[{kd[0]:+.2f}, {kd[1]:+.2f}, {kd[2]:+.2f}]",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

        if not target_found:
            cv2.putText(vis, f"ID{TARGET_ID} not detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        cv2.putText(vis, "q = quit", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("king_vector_publisher", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
