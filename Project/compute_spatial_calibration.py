#!/usr/bin/env python3
"""
Translation-only (point-cloud) hand-eye calibration via Kabsch-Umeyama.

Aligns 3D point pairs:
  - Marker position in CAMERA frame (from solvePnP)
  - EE position in BASE frame (from Franka)

Saves T_base_cam to spatial_calib/T_base_cam.npy so downstream scripts can
load it independently of any other calibration method.

Usage:
    python3 compute_spatial_calibration.py
"""
import os
import glob
import numpy as np
import cv2

DATA_DIR = "calib_data"
OUT_DIR = "spatial_calib"
MARKER_LEN = 0.04
MARKER_ID = 0

os.makedirs(OUT_DIR, exist_ok=True)

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
try:
    _det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    detect = lambda im: _det.detectMarkers(im)
except AttributeError:
    _p = cv2.aruco.DetectorParameters_create()
    detect = lambda im: cv2.aruco.detectMarkers(im, DICT, parameters=_p)

h = MARKER_LEN / 2.0
OBJ_PTS = np.array(
    [[-h,  h, 0], [ h,  h, 0], [ h, -h, 0], [-h, -h, 0]], dtype=np.float32
)


def main():
    K = np.load(f"{DATA_DIR}/K.npy")
    D = np.load(f"{DATA_DIR}/D.npy")

    pts_cam, pts_base = [], []
    for img_path in sorted(glob.glob(f"{DATA_DIR}/images/*.png")):
        idx = os.path.basename(img_path).split(".")[0]
        T_path = f"{DATA_DIR}/images/{idx}_T_base_ee.npy"
        if not os.path.exists(T_path):
            continue
        img = cv2.imread(img_path)
        corners, ids, _ = detect(img)
        if ids is None or MARKER_ID not in ids.flatten():
            continue
        i = np.where(ids.flatten() == MARKER_ID)[0][0]
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_PTS, corners[i][0].astype(np.float32), K, D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        pts_cam.append(tvec.flatten())
        pts_base.append(np.load(T_path)[:3, 3])

    pts_cam = np.array(pts_cam)
    pts_base = np.array(pts_base)
    print(f"Using {len(pts_cam)} point correspondences.")
    if len(pts_cam) < 3:
        raise RuntimeError("Need at least 3 correspondences.")

    # Kabsch-Umeyama: find R, t such that R @ pts_cam + t ≈ pts_base
    cc, cb = pts_cam.mean(axis=0), pts_base.mean(axis=0)
    H = (pts_cam - cc).T @ (pts_base - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ cc

    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R
    T_base_cam[:3, 3] = t

    print("\nT_base_cam (spatial / Kabsch-Umeyama) =")
    print(T_base_cam.round(4))
    print(f"\nCamera origin in base: {t.round(3)} m")

    pred = (R @ pts_cam.T).T + t
    err = np.linalg.norm(pred - pts_base, axis=1)
    print(f"\nResiduals (includes fixed marker-to-EE offset):")
    print(f"  mean = {err.mean():.4f} m   max = {err.max():.4f} m   std = {err.std():.4f} m")

    out_path = f"{OUT_DIR}/T_base_cam.npy"
    np.save(out_path, T_base_cam)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
