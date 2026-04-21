"""
Grasp planner for standing chess piece -- top-down approach.

Uses MuJoCo as an offline planning model for IK only.
Robot motion is handled by frankapy (point-to-point), so no RRT needed.

Coordinate frame convention (from perception):
  - piece_pos  = center of the piece (NOT body origin)
  - piece_quat = orientation with Z pointing up along the piece
  - X, Y axes  = arbitrary (don't matter -- stem is cylindrical)

The grasp point is computed as:
  grasp_center = piece_pos + R_piece @ [0, 0, grasp_z_offset]
where grasp_z_offset is the signed distance from the published center
to the desired grasp height on the stem.
"""
import numpy as np
import mujoco as mj


# ======================================================================
#  Quaternion / rotation helpers
# ======================================================================

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])


def rotmat_to_quat(R):
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
    return np.array([w, x, y, z])


def quat_error(q_current, q_target):
    q_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    q_err = quat_multiply(q_target, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


# ======================================================================
#  Grasp geometry -- corrected for center-of-piece frame
# ======================================================================

def calculate_grasp(piece_pos, piece_quat, cfg):
    """Compute top-down grasp targets for a standing piece.

    Args:
        piece_pos:  (3,) center of piece from perception
        piece_quat: (4,) [w,x,y,z] orientation -- Z up along piece
        cfg:        dict with grasp_z_offset_from_center, pre_grasp_height,
                    lift_height, gripper_open_width
    Returns:
        dict with grasp_center, finger_axis, gripper_quat, and all waypoints
    """
    R_piece = quat_to_rotmat(piece_quat)

    grasp_z_offset = cfg["grasp_z_offset_from_center"]
    grasp_center = piece_pos + R_piece @ np.array([0.0, 0.0, grasp_z_offset])

    # Finger approach axis -- piece X projected to horizontal
    finger_axis = R_piece @ np.array([1.0, 0.0, 0.0])
    finger_axis[2] = 0.0
    norm = np.linalg.norm(finger_axis)
    if norm < 1e-6:
        finger_axis = np.array([1.0, 0.0, 0.0])
    else:
        finger_axis = finger_axis / norm

    grip_w = cfg["gripper_open_width"]
    target_left = grasp_center + grip_w * finger_axis
    target_right = grasp_center - grip_w * finger_axis

    # Top-down gripper orientation: Z down, Y opposite to finger axis
    gripper_z = np.array([0.0, 0.0, -1.0])
    gripper_y = -finger_axis
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x = gripper_x / np.linalg.norm(gripper_x)
    gripper_y = np.cross(gripper_z, gripper_x)

    R_gripper = np.column_stack([gripper_x, gripper_y, gripper_z])
    gripper_quat = rotmat_to_quat(R_gripper)

    pre_h = cfg["pre_grasp_height"]
    lift_h = cfg["lift_height"]

    return {
        "grasp_center": grasp_center,
        "finger_axis": finger_axis,
        "target_left": target_left,
        "target_right": target_right,
        "gripper_quat": gripper_quat,
        "pre_grasp_tip": grasp_center + np.array([0, 0, pre_h]),
        "grasp_tip": grasp_center.copy(),
        "lift_tip": grasp_center + np.array([0, 0, lift_h]),
    }


def calculate_placement(place_pos, gripper_quat, cfg):
    """Compute placement waypoints. place_pos is the target (x,y,z)."""
    R_gripper = quat_to_rotmat(gripper_quat)
    gz = R_gripper[:, 2]
    offset = cfg["pre_place_offset"]

    place_tip = np.array(place_pos, dtype=float)
    pre_place_tip = place_tip - offset * gz + np.array([0, 0, 0.04])
    retreat_tip = place_tip - offset * gz + np.array([0, 0, 0.06])

    return {
        "place_quat": gripper_quat,
        "pre_place_tip": pre_place_tip,
        "place_tip": place_tip,
        "retreat_tip": retreat_tip,
    }


# ======================================================================
#  IK solver (uses MuJoCo planning model)
# ======================================================================

class GraspPlanner:
    """Wraps the MuJoCo-based IK solver. Motion planning is handled by
    frankapy, so no RRT is needed here."""

    def __init__(self, model_xml_path, cfg):
        self.cfg = cfg
        self.model = mj.MjModel.from_xml_path(model_xml_path)
        self.grasp_offset = cfg["grasp_offset"]
        self.ik_tol = cfg["ik_tol"]
        self.ik_max_iter = cfg["ik_max_iter"]
        self.grip_open = cfg["gripper_open_width"]

        self.hand_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.elbow_body_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, n)
            for n in ("link3", "link4", "link5")
        ]

    def compute_ik(self, target_pos, target_quat, q_init, min_elbow_z=None):
        model = self.model
        nv = 7
        ik_step = 0.3
        ik_damping = 1e-4

        d = mj.MjData(model)
        d.qpos[:7] = q_init.copy()
        d.qpos[7] = self.grip_open
        d.qpos[8] = self.grip_open
        mj.mj_forward(model, d)

        for _ in range(self.ik_max_iter):
            hand_pos = d.xpos[self.hand_body_id].copy()
            hand_quat = d.xquat[self.hand_body_id].copy()
            hand_rot = d.xmat[self.hand_body_id].reshape(3, 3)
            hand_z = hand_rot[:, 2]
            tip_pos = hand_pos + self.grasp_offset * hand_z

            pos_err = target_pos - tip_pos
            ori_err = quat_error(hand_quat, target_quat)
            err = np.concatenate([pos_err, ori_err])

            if (np.linalg.norm(err[:3]) < self.ik_tol and
                    np.linalg.norm(err[3:]) < self.ik_tol * 5):
                return d.qpos[:7].copy(), True

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacBody(model, d, jacp, jacr, self.hand_body_id)

            Jp = jacp[:, :nv]
            Jr = jacr[:, :nv]

            ow = self.grasp_offset * hand_z
            skew = np.array([
                [0, -ow[2], ow[1]],
                [ow[2], 0, -ow[0]],
                [-ow[1], ow[0], 0],
            ])
            J = np.vstack([Jp - skew @ Jr, Jr])

            JJT = J @ J.T + ik_damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, err)

            if min_elbow_z is not None:
                J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))
                N = np.eye(nv) - J_pinv @ J
                grad = np.zeros(nv)
                for eid in self.elbow_body_ids:
                    z = d.xpos[eid][2]
                    if z < min_elbow_z:
                        jac_e = np.zeros((3, model.nv))
                        mj.mj_jacBody(model, d, jac_e, None, eid)
                        grad += jac_e[2, :nv] * (min_elbow_z - z)
                dq += N @ grad * 2.0

            d.qpos[:nv] += ik_step * dq
            for j in range(nv):
                lo, hi = model.jnt_range[j]
                if lo < hi:
                    d.qpos[j] = np.clip(d.qpos[j], lo, hi)
            mj.mj_forward(model, d)

        return d.qpos[:7].copy(), False

    # -- Full plan --

    def plan_full_grasp(self, piece_pos, piece_quat, place_pos):
        """Compute IK for the full pick-and-place sequence.

        Args:
            piece_pos:  (3,) center of piece from perception
            piece_quat: (4,) [w,x,y,z] from perception (Z up along piece)
            place_pos:  (3,) target placement position

        Returns:
            dict with joint configs and success flag
        """
        cfg = self.cfg

        grasp = calculate_grasp(piece_pos, piece_quat, cfg)
        place = calculate_placement(place_pos, grasp["gripper_quat"], cfg)

        home_q = np.array(cfg["home_q"], dtype=float)
        table_h = cfg["table_height"]
        min_z = table_h + cfg["collision_margin"] + 0.02

        gq = grasp["gripper_quat"]
        pq = place["place_quat"]

        q_pre, ok1 = self.compute_ik(
            grasp["pre_grasp_tip"], gq, home_q, min_z)
        q_grasp, ok2 = self.compute_ik(
            grasp["grasp_tip"], gq, q_pre, min_z)
        q_lift, ok3 = self.compute_ik(
            grasp["lift_tip"], gq, q_grasp, min_z)
        q_pre_place, ok4 = self.compute_ik(
            place["pre_place_tip"], pq, q_lift, min_z)
        q_place, ok5 = self.compute_ik(
            place["place_tip"], pq, q_pre_place, min_z)
        q_retreat, ok6 = self.compute_ik(
            place["retreat_tip"], pq, q_place, min_z)

        ik_ok = all([ok1, ok2, ok3, ok4, ok5, ok6])

        return {
            "ik_ok": ik_ok,
            "ik_results": [ok1, ok2, ok3, ok4, ok5, ok6],
            "joint_configs": {
                "home": home_q,
                "pre_grasp": q_pre,
                "grasp": q_grasp,
                "lift": q_lift,
                "pre_place": q_pre_place,
                "place": q_place,
                "retreat": q_retreat,
            },
            "grasp_data": grasp,
            "place_data": place,
        }
