#!/usr/bin/env python3
"""
Hand-eye calibration DATA COLLECTION.

Run INSIDE the frankapy docker container on the User PC:
    bash terminal_docker.sh
    python3 collect_calib_data.py

Prerequisites:
  - roscore running on the User PC
  - realsense2_camera driver running (roslaunch realsense2_camera rs_camera.launch)
  - frankapy control PC server started (start_control_pc.sh)
  - Robot unlocked and in blue (program) mode

Usage:
  - A window pops up showing the live RealSense feed.
  - The arm is put in guide mode so you can move it freely by hand.
  - Move the arm to a pose where the marker is clearly in view of the camera,
    then press 's' in the window to save the image + EE pose pair.
  - Repeat for ~15-20 poses with GOOD ROTATIONAL VARIETY (not just translations).
  - Press 'q' to quit when done.

Output:
  calib_data/
    K.npy, D.npy           -- camera intrinsics + distortion
    images/000.png         -- RGB image at pose 0
    images/000_T_base_ee.npy -- 4x4 end-effector pose in robot base frame
    images/001.png ...
"""
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from frankapy import FrankaArm

SAVE_DIR = "calib_data"
os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)

bridge = CvBridge()
latest_image = None


def image_cb(msg):
    global latest_image
    latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")


def main():
    # FrankaArm() calls rospy.init_node() internally -- must come FIRST and
    # we must NOT call rospy.init_node() ourselves.
    fa = FrankaArm()

    rospy.Subscriber("/camera/color/image_raw", Image, image_cb)

    # Grab intrinsics once
    print("Waiting for /camera/color/camera_info ...")
    K_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=5.0)
    K = np.array(K_msg.K).reshape(3, 3)
    D = np.array(K_msg.D)
    np.save(f"{SAVE_DIR}/K.npy", K)
    np.save(f"{SAVE_DIR}/D.npy", D)
    print(f"Saved intrinsics.\nK =\n{K}\nD = {D}")

    fa.run_guide_mode(duration=1e6, block=False)
    print("\nGuide mode ENABLED. Move the arm by hand.")
    print("Focus the 'capture' window, then: 's' = save pair,  'q' = quit\n")

    i = 0
    while not rospy.is_shutdown():
        if latest_image is None:
            rospy.sleep(0.05)
            continue
        disp = latest_image.copy()
        cv2.putText(
            disp,
            f"saved: {i}  [s]=save  [q]=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("capture", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            img_path = f"{SAVE_DIR}/images/{i:03d}.png"
            cv2.imwrite(img_path, latest_image)
            T = fa.get_pose().matrix  # 4x4 T_base_ee
            np.save(f"{SAVE_DIR}/images/{i:03d}_T_base_ee.npy", T)
            print(f"[{i}] saved  |  translation = {T[:3, 3].round(3)}")
            i += 1

    cv2.destroyAllWindows()
    fa.stop_skill()
    print(f"\nDone. Collected {i} pose/image pairs in {SAVE_DIR}/")
    if i < 10:
        print("WARNING: fewer than 10 poses. Calibration will be poor.")


if __name__ == "__main__":
    main()
