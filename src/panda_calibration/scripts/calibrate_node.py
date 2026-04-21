#!/usr/bin/env python
"""
Eye-to-hand camera calibration for Franka Panda + RealSense.

Two ArUco markers (ID 0 left, ID 1 right) are attached to the end effector.
The robot is moved to various poses. At each pose, publish to /calibrate/capture
to record a sample (EE pose from frankapy + marker pose from camera). After
enough samples, publish to /calibrate/solve to compute T_camera_to_base.

EE pose source: frankapy (fa.get_pose()) — no franka_control / TF needed.

Subscribes:
  /camera/color/image_raw    (sensor_msgs/Image)
  /camera/color/camera_info  (sensor_msgs/CameraInfo)
  /calibrate/capture         (std_msgs/Empty)     — take one sample
  /calibrate/solve           (std_msgs/Empty)     — run calibration

Publishes:
  /calibration/image          (sensor_msgs/Image)       — annotated camera view
  /calibration/markers        (visualization_msgs/MarkerArray) — detected markers in RViz
  /calibration/status         (std_msgs/String)         — current status
  /tf_static                  (after solve)             — panda_link0 → camera_color_optical_frame
"""
import threading
import os
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco

import rospy
import tf2_ros
from cv_bridge import CvBridge
from std_msgs.msg import Empty, String, Header, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (
    Point, Quaternion, TransformStamped, Vector3,
)
from visualization_msgs.msg import Marker, MarkerArray

from frankapy import FrankaArm


ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
}

CALIB_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def rotmat_to_quat(R):
    """3x3 rotation matrix -> [x, y, z, w] quaternion (ROS convention)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w])


class CalibrateNode:
    def __init__(self):
        rospy.init_node("panda_calibration")

        # ── Params ──
        dict_name = rospy.get_param("~aruco_dict", "DICT_5X5_100")
        self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        self.aruco_params = aruco.DetectorParameters()
        self.marker_size = rospy.get_param("~marker_size", 0.05)
        self.id_left = rospy.get_param("~marker_id_left", 0)
        self.id_right = rospy.get_param("~marker_id_right", 1)
        self.min_samples = rospy.get_param("~min_samples", 8)
        method_name = rospy.get_param("~method", "TSAI")
        self.method = CALIB_METHODS.get(method_name, cv2.CALIB_HAND_EYE_TSAI)
        self.output_file = rospy.get_param("~output_file", "calibration_result.yaml")

        # ── Frankapy ──
        rospy.loginfo("Connecting to FrankaArm...")
        self.fa = FrankaArm()
        rospy.loginfo("FrankaArm connected.")

        # ── State ──
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.lock = threading.Lock()
        self.samples = []
        self.T_cam_to_base = None

        # ── TF (for publishing result only) ──
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # ── Subscribers ──
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        rospy.Subscriber(image_topic, Image, self._image_cb)
        rospy.Subscriber(info_topic, CameraInfo, self._info_cb)
        rospy.Subscriber("/calibrate/capture", Empty, self._capture_cb)
        rospy.Subscriber("/calibrate/solve", Empty, self._solve_cb)

        # ── Publishers ──
        self.image_pub = rospy.Publisher(
            "/calibration/image", Image, queue_size=1)
        self.marker_pub = rospy.Publisher(
            "/calibration/markers", MarkerArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            "/calibration/status", String, queue_size=1, latch=True)

        self._publish_status("Ready. 0 samples. Publish /calibrate/capture to add.")
        rospy.loginfo("Calibration node ready.")
        rospy.loginfo("  Markers: ID %d (left), ID %d (right), size=%.3fm",
                      self.id_left, self.id_right, self.marker_size)

    # ── EE pose from frankapy ──────────────────────────────────

    def _get_ee_pose(self):
        """Read current EE pose from frankapy as 4x4 matrix."""
        try:
            pose = self.fa.get_pose()
            T = np.eye(4)
            T[:3, :3] = pose.rotation
            T[:3, 3] = pose.translation
            return T
        except Exception as e:
            rospy.logwarn("Failed to read EE pose: %s", e)
            return None

    # ── Callbacks ──────────────────────────────────────────────

    def _info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics received.")

    def _image_cb(self, msg):
        with self.lock:
            self.latest_image = msg
        self._detect_and_publish(msg)

    def _capture_cb(self, msg):
        if self.camera_matrix is None:
            rospy.logwarn("No camera intrinsics yet.")
            return

        T_base_ee = self._get_ee_pose()
        if T_base_ee is None:
            return

        with self.lock:
            img_msg = self.latest_image
        if img_msg is None:
            rospy.logwarn("No camera image yet.")
            return

        detections = self._detect_markers(img_msg)
        if not detections:
            rospy.logwarn("No ArUco markers detected — not captured.")
            return

        T_cam_marker = None
        used_id = None
        for mid in [self.id_left, self.id_right]:
            if mid in detections:
                T_cam_marker = detections[mid]
                used_id = mid
                break

        if T_cam_marker is None:
            rospy.logwarn("Neither marker %d nor %d detected.", self.id_left, self.id_right)
            return

        self.samples.append((T_base_ee, T_cam_marker))
        n = len(self.samples)
        rospy.loginfo("Sample %d captured (marker ID %d). EE: [%.3f, %.3f, %.3f]",
                      n, used_id, *T_base_ee[:3, 3])
        self._publish_status(
            "%d samples. Need %d min. Publish /calibrate/solve when ready." %
            (n, self.min_samples))

    def _solve_cb(self, msg):
        n = len(self.samples)
        if n < self.min_samples:
            rospy.logwarn("Only %d samples — need %d.", n, self.min_samples)
            return

        rospy.loginfo("Solving with %d samples...", n)

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_base_ee, T_cam_marker in self.samples:
            T_ee_base = np.linalg.inv(T_base_ee)
            R_gripper2base.append(T_ee_base[:3, :3])
            t_gripper2base.append(T_ee_base[:3, 3].reshape(3, 1))
            R_target2cam.append(T_cam_marker[:3, :3])
            t_target2cam.append(T_cam_marker[:3, 3].reshape(3, 1))

        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=self.method,
        )

        self.T_cam_to_base = np.eye(4)
        self.T_cam_to_base[:3, :3] = R_cam2base
        self.T_cam_to_base[:3, 3] = t_cam2base.flatten()

        errors = []
        for T_base_ee, T_cam_marker in self.samples:
            T_est = T_cam_marker @ np.linalg.inv(T_base_ee)
            errors.append(np.linalg.norm(T_est[:3, 3] - self.T_cam_to_base[:3, 3]))
        mean_err = np.mean(errors) * 1000
        max_err = np.max(errors) * 1000

        rospy.loginfo("=== CALIBRATION RESULT ===")
        rospy.loginfo("T_camera_to_base:\n%s",
                      np.array2string(self.T_cam_to_base, precision=6))
        rospy.loginfo("Translation: [%.4f, %.4f, %.4f]", *self.T_cam_to_base[:3, 3])
        q = rotmat_to_quat(self.T_cam_to_base[:3, :3])
        rospy.loginfo("Quaternion [x,y,z,w]: [%.6f, %.6f, %.6f, %.6f]", *q)
        rospy.loginfo("Consistency: mean=%.1fmm, max=%.1fmm", mean_err, max_err)

        self._publish_calibration_tf()
        self._save_calibration()
        self._publish_status(
            "CALIBRATED (%d samples). Error: mean=%.1fmm max=%.1fmm" %
            (n, mean_err, max_err))

    # ── ArUco detection ────────────────────────────────────────

    def _detect_markers(self, img_msg):
        if self.camera_matrix is None:
            return {}

        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return {}

        half = self.marker_size / 2.0
        obj_pts = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)

        results = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in (self.id_left, self.id_right):
                continue
            img_pts = corners[i].reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            results[marker_id] = T
        return results

    def _detect_and_publish(self, img_msg):
        if self.camera_matrix is None:
            return

        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        vis = cv_img.copy()
        aruco.drawDetectedMarkers(vis, corners, ids)

        if ids is not None:
            half = self.marker_size / 2.0
            obj_pts = np.array([
                [-half, half, 0], [half, half, 0],
                [half, -half, 0], [-half, -half, 0],
            ], dtype=np.float32)

            ma = MarkerArray()
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in (self.id_left, self.id_right):
                    continue
                img_pts = corners[i].reshape(4, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.camera_matrix, self.dist_coeffs)
                if not ok:
                    continue

                cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, self.marker_size * 0.6)

                c = corners[i][0][0]
                label = "LEFT(ID%d)" % marker_id if marker_id == self.id_left else "RIGHT(ID%d)" % marker_id
                cv2.putText(vis, label, (int(c[0]), int(c[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                R, _ = cv2.Rodrigues(rvec)
                q = rotmat_to_quat(R)
                m = Marker()
                m.header = Header(stamp=rospy.Time.now(),
                                  frame_id="camera_color_optical_frame")
                m.ns = "aruco"
                m.id = int(marker_id)
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position = Point(*tvec.flatten())
                m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                m.scale.x = self.marker_size
                m.scale.y = self.marker_size
                m.scale.z = 0.002
                color = (0, 1, 0) if marker_id == self.id_left else (0, 0.5, 1)
                m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.7)
                m.lifetime = rospy.Duration(0.5)
                ma.markers.append(m)

            self.marker_pub.publish(ma)

        n = len(self.samples)
        status = "Samples: %d" % n
        if self.T_cam_to_base is not None:
            status += " | CALIBRATED"
        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    # ── Output ─────────────────────────────────────────────────

    def _publish_calibration_tf(self):
        if self.T_cam_to_base is None:
            return
        T_base_to_cam = np.linalg.inv(self.T_cam_to_base)
        q = rotmat_to_quat(T_base_to_cam[:3, :3])

        ts = TransformStamped()
        ts.header.stamp = rospy.Time.now()
        ts.header.frame_id = "panda_link0"
        ts.child_frame_id = "camera_color_optical_frame"
        ts.transform.translation = Vector3(*T_base_to_cam[:3, 3])
        ts.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.static_broadcaster.sendTransform(ts)
        rospy.loginfo("Published static TF: panda_link0 -> camera_color_optical_frame")

    def _save_calibration(self):
        if self.T_cam_to_base is None:
            return
        T = self.T_cam_to_base
        q = rotmat_to_quat(T[:3, :3])
        data = {
            "T_camera_to_base": {
                "translation": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])},
                "rotation_quaternion_xyzw": {
                    "x": float(q[0]), "y": float(q[1]),
                    "z": float(q[2]), "w": float(q[3]),
                },
                "rotation_matrix": T[:3, :3].tolist(),
            },
            "num_samples": len(self.samples),
        }
        path = self.output_file
        if not os.path.isabs(path):
            path = os.path.join(os.path.expanduser("~"), path)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        rospy.loginfo("Saved to %s", path)

    def _publish_status(self, text):
        self.status_pub.publish(String(data=text))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = CalibrateNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
