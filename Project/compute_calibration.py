#!/usr/bin/env python3
"""
Hand-eye calibration COMPUTATION.

Run on the User PC (inside or outside docker; only needs numpy + opencv) AFTER
you've collected data with collect_calib_data.py.

    python3 compute_calibration.py

Inputs:
  calib_data/K.npy, calib_data/D.npy        -- intrinsics saved by collector
  calib_data/images/*.png                   -- RGB images
  calib_data/images/*_T_base_ee.npy         -- corresponding EE poses

Output:
  calib_data/T_base_cam.npy  -- 4x4 extrinsics (camera frame -> base frame)
  Prints a residual sanity check (std should be < a few mm if data is good)

Later use:
    T_base_cam = np.load("calib_data/T_base_cam.npy")
    T_base_object = T_base_cam @ T_cam_object
"""
import os
import glob
import numpy as np
import cv2

DATA_DIR = "calib_data"
MARKER_LEN = 0.04  # 4 cm marker
MARKER_ID = 0

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Detector API differs between OpenCV < 4.7 and >= 4.7
try:
    _detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    def detect(img):
        return _detector.detectMarkers(img)
except AttributeError:
    _params = cv2.aruco.DetectorParameters_create()
    def detect(img):
        return cv2.aruco.detectMarkers(img, DICT, parameters=_params)


def main():
    K = np.load(f"{DATA_DIR}/K.npy")
    D = np.load(f"{DATA_DIR}/D.npy")

    # Marker corners in marker frame (z=0 plane).
    # Order must match what detectMarkers returns: TL, TR, BR, BL.
    h = MARKER_LEN / 2.0
    obj_pts = np.array(
        [[-h,  h, 0],
         [ h,  h, 0],
         [ h, -h, 0],
         [-h, -h, 0]], dtype=np.float32,
    )

    R_g2b, t_g2b = [], []   # gripper-to-base (we'll feed INVERTED EE pose)
    R_t2c, t_t2c = [], []   # target(marker)-to-camera

    for img_path in sorted(glob.glob(f"{DATA_DIR}/images/*.png")):
        idx = os.path.basename(img_path).split(".")[0]
        T_path = f"{DATA_DIR}/images/{idx}_T_base_ee.npy"
        if not os.path.exists(T_path):
            continue

        T_base_ee = np.load(T_path)
        img = cv2.imread(img_path)
        corners, ids, _ = detect(img)
        if ids is None or MARKER_ID not in ids.flatten():
            print(f"[{idx}] marker {MARKER_ID} not found -- skipping")
            continue

        i = np.where(ids.flatten() == MARKER_ID)[0][0]
        img_pts = corners[i][0].astype(np.float32)  # 4x2

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            print(f"[{idx}] solvePnP failed -- skipping")
            continue

        R_marker, _ = cv2.Rodrigues(rvec)
        R_t2c.append(R_marker)
        t_t2c.append(tvec.flatten())

        # Eye-to-hand trick: invert EE pose so the output is T_base_cam
        T_ee_base = np.linalg.inv(T_base_ee)
        R_g2b.append(T_ee_base[:3, :3])
        t_g2b.append(T_ee_base[:3, 3])

    print(f"\nUsing {len(R_g2b)} valid pose pairs.")
    if len(R_g2b) < 3:
        raise RuntimeError("Need at least 3 (ideally 15+) valid pose pairs.")

    methods = [
        ("TSAI",        cv2.CALIB_HAND_EYE_TSAI),
        ("PARK",        cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD",      cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF",     cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS",  cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]

    best = None  # (std_sum, name, T_base_cam)
    for name, flag in methods:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=flag
        )
        T_bc = np.eye(4)
        T_bc[:3, :3] = R_cam2base
        T_bc[:3, 3] = t_cam2base.flatten()
        # Quick residual: std of reconstructed T_ee_marker translation
        offs = []
        for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
            T_eb = np.eye(4); T_eb[:3,:3]=Rg; T_eb[:3,3]=tg
            T_cm = np.eye(4); T_cm[:3,:3]=Rt; T_cm[:3,3]=tt
            offs.append((T_eb @ T_bc @ T_cm)[:3, 3])
        s = np.array(offs).std(axis=0).sum()
        print(f"  {name:12s}  t={t_cam2base.flatten().round(3)}  "
              f"residual std sum = {s:.4f}")
        if best is None or s < best[0]:
            best = (s, name, T_bc)

    print(f"\nBest method: {best[1]}  (residual std sum = {best[0]:.4f})")
    T_base_cam = best[2]

    print("\nT_base_cam =")
    print(np.round(T_base_cam, 4))
    print(f"\nCamera origin in robot base frame: "
          f"{T_base_cam[:3, 3].round(3)} m")

    np.save(f"{DATA_DIR}/T_base_cam.npy", T_base_cam)
    print(f"Saved to {DATA_DIR}/T_base_cam.npy")

    # --- Residual sanity check ---
    # The marker is rigidly attached to the EE, so T_ee_marker should be the
    # SAME for every pose. We can reconstruct it as:
    #   T_ee_marker_i = T_ee_base · T_base_cam · T_cam_marker_i
    # and check how much it varies across poses.
    offsets = []
    for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        T_eb = np.eye(4); T_eb[:3, :3] = Rg;  T_eb[:3, 3] = tg     # T_ee_base
        T_cm = np.eye(4); T_cm[:3, :3] = Rt;  T_cm[:3, 3] = tt     # T_cam_marker
        T_ee_marker = T_eb @ T_base_cam @ T_cm
        offsets.append(T_ee_marker[:3, 3])
    offsets = np.array(offsets)
    print(f"\nMarker-in-EE translation across poses (should be ~constant):")
    print(f"  mean   = {offsets.mean(axis=0).round(4)} m")
    print(f"  stddev = {offsets.std(axis=0).round(4)} m  "
          f"(good: < 0.003, decent: < 0.006)")


if __name__ == "__main__":
    main()
