"""
Grasp planner for standing chess piece — top-down approach.

Uses MuJoCo as an offline planning model for IK and collision-free RRT.
Coordinate frame convention (from perception):
  - piece_pos  = center of the piece (NOT body origin)
  - piece_quat = orientation with Z pointing up along the piece
  - X, Y axes  = arbitrary (don't matter — stem is cylindrical)

The grasp point is computed as:
  grasp_center = piece_pos + R_piece @ [0, 0, grasp_z_offset]
where grasp_z_offset is the signed distance from the published center
to the desired grasp height on the stem.
"""
import numpy as np
import mujoco as mj


# ══════════════════════════════════════════════════════════════
#  Quaternion / rotation helpers
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  Grasp geometry — corrected for center-of-piece frame
# ══════════════════════════════════════════════════════════════

def calculate_grasp(piece_pos, piece_quat, cfg):
    """Compute top-down grasp targets for a standing piece.

    Args:
        piece_pos:  (3,) center of piece from perception
        piece_quat: (4,) [w,x,y,z] orientation — Z up along piece
        cfg:        dict with grasp_z_offset_from_center, pre_grasp_height,
                    lift_height, gripper_open_width
    Returns:
        dict with grasp_center, finger_axis, gripper_quat, and all waypoints
    """
    R_piece = quat_to_rotmat(piece_quat)

    # Grasp point = piece center + offset along the piece Z axis
    # Since Z goes up the piece, a positive offset moves toward the top
    grasp_z_offset = cfg["grasp_z_offset_from_center"]
    grasp_center = piece_pos + R_piece @ np.array([0.0, 0.0, grasp_z_offset])

    # Finger approach axis — piece X projected to horizontal
    # (arbitrary because stem is cylindrical; just pick one)
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


# ══════════════════════════════════════════════════════════════
#  IK solver (uses MuJoCo planning model)
# ══════════════════════════════════════════════════════════════

class GraspPlanner:
    """Wraps the MuJoCo-based IK solver and RRT planner."""

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

    # ── RRT planner ──────────────────────────────────────────

    def plan_motion(self, q_start, q_goal, gripper_val, piece_qpos,
                    max_iter=5000, step_size=0.15, goal_bias=0.2,
                    smooth_iter=200):
        model = self.model
        check_data = mj.MjData(model)

        table_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_0")
        ground_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
        obstacles = frozenset({table_id, ground_id})

        piece_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ChessPiece_0")
        link0_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link0")
        skip_bodies = frozenset({piece_body, 0, link0_body})

        joint_lo = np.array([model.jnt_range[i, 0] for i in range(7)])
        joint_hi = np.array([model.jnt_range[i, 1] for i in range(7)])
        n_piece = len(piece_qpos)

        def col_free(q):
            check_data.qpos[:7] = q
            check_data.qpos[7] = gripper_val
            check_data.qpos[8] = gripper_val
            check_data.qpos[9:9 + n_piece] = piece_qpos
            mj.mj_forward(model, check_data)
            for ci in range(check_data.ncon):
                c = check_data.contact[ci]
                if c.geom1 not in obstacles and c.geom2 not in obstacles:
                    continue
                other = c.geom2 if c.geom1 in obstacles else c.geom1
                if model.geom_bodyid[other] in skip_bodies:
                    continue
                return False
            return True

        def edge_free(q1, q2, n=10):
            for i in range(n + 1):
                if not col_free(q1 + (i / n) * (q2 - q1)):
                    return False
            return True

        if edge_free(q_start, q_goal, n=20):
            return self._interpolate(q_start, q_goal, 60)

        if not col_free(q_start):
            return self._interpolate(q_start, q_goal, 60)
        if not col_free(q_goal):
            return self._interpolate(q_start, q_goal, 60)

        nodes = [q_start.copy()]
        parents = [-1]
        path = None

        for _ in range(max_iter):
            q_rand = (q_goal.copy() if np.random.random() < goal_bias
                      else np.random.uniform(joint_lo, joint_hi))

            dists = np.linalg.norm(np.array(nodes) - q_rand, axis=1)
            nearest_idx = int(np.argmin(dists))
            q_nearest = nodes[nearest_idx]

            diff = q_rand - q_nearest
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            q_new = (q_nearest + step_size * diff / dist
                     if dist > step_size else q_rand.copy())
            q_new = np.clip(q_new, joint_lo, joint_hi)

            if not edge_free(q_nearest, q_new):
                continue

            nodes.append(q_new.copy())
            parents.append(nearest_idx)

            if np.linalg.norm(q_new - q_goal) < step_size:
                if edge_free(q_new, q_goal):
                    nodes.append(q_goal.copy())
                    parents.append(len(nodes) - 2)
                    idx = len(nodes) - 1
                    path = []
                    while idx != -1:
                        path.append(nodes[idx])
                        idx = parents[idx]
                    path.reverse()
                    break

        if path is None:
            return self._interpolate(q_start, q_goal, 60)

        for _ in range(smooth_iter):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 1)
            j = np.random.randint(i + 2, len(path))
            if edge_free(path[i], path[j]):
                path = path[:i + 1] + path[j:]

        segs = []
        for i in range(len(path) - 1):
            d = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
            n_pts = max(10, int(d / 0.02))
            segs.append(self._interpolate(
                np.array(path[i]), np.array(path[i + 1]), n_pts))
        return np.vstack(segs)

    @staticmethod
    def _interpolate(q_start, q_end, n):
        traj = np.zeros((n, 7))
        for i in range(n):
            t = i / max(n - 1, 1)
            t = 3 * t**2 - 2 * t**3
            traj[i] = q_start + t * (q_end - q_start)
        return traj

    # ── Full plan ────────────────────────────────────────────

    def plan_full_grasp(self, piece_pos, piece_quat, place_pos):
        """Compute IK + trajectories for the full pick-and-place sequence.

        Args:
            piece_pos:  (3,) center of piece from perception
            piece_quat: (4,) [w,x,y,z] from perception (Z up along piece)
            place_pos:  (3,) target placement position

        Returns:
            dict with joint configs, trajectories, and success flag
        """
        cfg = self.cfg

        grasp = calculate_grasp(piece_pos, piece_quat, cfg)
        place = calculate_placement(place_pos, grasp["gripper_quat"], cfg)

        home_q = np.array(cfg["home_q"], dtype=float)
        table_h = cfg["table_height"]
        min_z = table_h + cfg["collision_margin"] + 0.02

        # Inflate table for planning
        table_gid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_GEOM, "table_0")
        orig_size = self.model.geom_size[table_gid].copy()
        self.model.geom_size[table_gid][2] += cfg["collision_margin"]

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

        # Build a dummy piece qpos for the planner's collision model.
        # Place piece at the perceived location with standing orientation.
        piece_qpos = np.concatenate([piece_pos, piece_quat])

        grip_open = cfg["gripper_open_width"]
        grip_closed = cfg["gripper_close_width"]

        trajs = {
            "to_home": self._interpolate(
                np.array(cfg["home_q"]), home_q, 60),
            "to_pre_grasp": self.plan_motion(
                home_q, q_pre, grip_open, piece_qpos),
            "to_grasp": self.plan_motion(
                q_pre, q_grasp, grip_open, piece_qpos),
            "to_lift": self.plan_motion(
                q_grasp, q_lift, grip_closed, piece_qpos),
            "to_pre_place": self.plan_motion(
                q_lift, q_pre_place, grip_closed, piece_qpos),
            "to_place": self.plan_motion(
                q_pre_place, q_place, grip_closed, piece_qpos),
            "to_retreat": self.plan_motion(
                q_place, q_retreat, grip_open, piece_qpos),
        }

        # Restore table
        self.model.geom_size[table_gid] = orig_size

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
            "trajs": trajs,
            "grasp_data": grasp,
            "place_data": place,
        }
