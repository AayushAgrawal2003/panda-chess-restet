"""
IK-based grasp-and-place for 4 chess pieces on 4 tables around a Franka Panda.

Each piece spawns laying down at a random (x, y, yaw) on its table.
The robot picks each piece, rotates it upright, and places it at a random
target location on the same table. Collision-free motion via RRT.

Workflow per piece (auto-advances with 0.5s pause):
  HOME_i → PRE-GRASP → GRASP → CLOSE → LIFT
         → PRE-PLACE → PLACE → OPEN → RETREAT → next table

Markers:
  RED cylinder = Drop target on table surface (always shown)
  Debug markers behind --markers flag.
"""
import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import argparse
import sys

# ── Scene config ──────────────────────────────────────────────
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
OUTPUT_XML = "franka_emika_panda/grasp_ik_scene.xml"

BASE_HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float)

TABLE_HEIGHT = 0.3
TABLE_HALF = 0.15            # 0.3m × 0.3m square tables
COLLISION_MARGIN = 0.05

# Grasp geometry
GRASP_OFFSET = 0.103
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.12

# Placement
PLACE_MIN_DIST = 0.08
PLACE_TIP_HEIGHT = TABLE_HEIGHT + 0.2
PRE_PLACE_OFFSET = 0.10

# Timing
PAUSE_SECONDS = 0.5

# IK parameters
IK_TOL = 1e-3
IK_MAX_ITER = 200
IK_STEP = 0.3
IK_DAMPING = 1e-4

# Spawn config
SPAWN_OFFSET = 0.08          # ±offset from table center
SPAWN_Z = TABLE_HEIGHT + 0.032
SPAWN_YAW_RANGE = (-1, 1)
PLACE_OFFSET = 0.08

NUM_TABLES = 4

# 4 tables around the robot — all within reachable workspace
TABLES = [
    {"name": "front",     "cx":  0.45, "cy":  0.00},
    {"name": "right",     "cx":  0.00, "cy": -0.45},
    {"name": "left",      "cx":  0.00, "cy":  0.45},
    {"name": "back_left", "cx": -0.32, "cy":  0.32},
]


def home_q_for_table(table):
    """HOME_Q with joint-1 rotated to face the table."""
    q = BASE_HOME_Q.copy()
    q[0] = np.arctan2(table["cy"], table["cx"])
    return q


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
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
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
    q_cur_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    q_err = quat_multiply(q_target, q_cur_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def axis_angle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    ha = angle / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


# ══════════════════════════════════════════════════════════════
#  Scene building
# ══════════════════════════════════════════════════════════════

def random_piece_pose(table):
    """Generate random (pos, quat, yaw) for piece laying on its side on the given table."""
    cx, cy = table["cx"], table["cy"]
    x = np.random.uniform(cx - SPAWN_OFFSET, cx + SPAWN_OFFSET)
    y = np.random.uniform(cy - SPAWN_OFFSET, cy + SPAWN_OFFSET)
    yaw = np.random.uniform(*SPAWN_YAW_RANGE)

    q_lay = np.array([0.7071068, 0, 0.7071068, 0])
    q_yaw = axis_angle_to_quat(np.array([0, 0, 1]), yaw)
    q_final = quat_multiply(q_yaw, q_lay)

    return [x, y, SPAWN_Z], q_final, yaw


def random_place_pos(table, grasp_xy):
    """Pick a random placement (x, y) on the given table, away from the grasp."""
    cx, cy = table["cx"], table["cy"]
    for _ in range(100):
        x = np.random.uniform(cx - PLACE_OFFSET, cx + PLACE_OFFSET)
        y = np.random.uniform(cy - PLACE_OFFSET, cy + PLACE_OFFSET)
        dist = np.linalg.norm(np.array([x, y]) - grasp_xy)
        if dist >= PLACE_MIN_DIST:
            return x, y
    return cx, cy


def build_scene(piece_poses):
    """Build XML scene with 4 tables and 4 chess pieces."""
    modelTree = ET.parse(ROOT_MODEL_XML)
    root = modelTree.getroot()

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {
        "name": "chess_king_mesh",
        "file": "chess_king.stl",
        "scale": "0.001 0.001 0.001"
    })

    worldbody = root.find("worldbody")

    # ── Tables ──
    table_half_h = TABLE_HEIGHT / 2.0
    colors = [
        "0.4 0.3 0.2 1",   # front  – dark wood
        "0.3 0.4 0.3 1",   # right  – green tint
        "0.3 0.3 0.4 1",   # left   – blue tint
        "0.4 0.35 0.25 1", # back   – light wood
    ]
    for i, table in enumerate(TABLES):
        ET.SubElement(worldbody, "geom", {
            "name": f"table_{i}",
            "type": "box",
            "size": f"{TABLE_HALF} {TABLE_HALF} {table_half_h}",
            "pos": f"{table['cx']} {table['cy']} {table_half_h}",
            "rgba": colors[i],
            "friction": "1.5 0.5 0.1",
            "contype": "1", "conaffinity": "1"
        })

    # ── Chess pieces ──
    for i, (pos, quat) in enumerate(piece_poses):
        body = ET.SubElement(worldbody, "body", {
            "name": f"ChessPiece_{i}",
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
        })
        ET.SubElement(body, "freejoint", {"name": f"chess_piece_joint_{i}"})

        ET.SubElement(body, "geom", {
            "name": f"chess_piece_visual_{i}",
            "type": "mesh", "mesh": "chess_king_mesh",
            "rgba": "0.85 0.75 0.55 1",
            "contype": "0", "conaffinity": "0", "mass": "0"
        })
        ET.SubElement(body, "geom", {
            "name": f"chess_piece_base_col_{i}",
            "type": "box",
            "size": "0.016 0.016 0.016",
            "pos": "0.016 0.016 0.016",
            "rgba": "0.85 0.75 0.55 0.3",
            "density": "500",
            "friction": "1.5 1.5 0.5",
            "contype": "1", "conaffinity": "1", "condim": "4"
        })
        ET.SubElement(body, "geom", {
            "name": f"chess_piece_stem_col_{i}",
            "type": "cylinder",
            "size": "0.010 0.030",
            "pos": "0.016 0.016 0.060",
            "rgba": "0.85 0.75 0.55 0.2",
            "density": "500",
            "friction": "1.5 0.5 0.1",
            "contype": "1", "conaffinity": "1", "condim": "4"
        })

    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


def add_markers_to_scene(place_positions, show_markers=False,
                         grasp_data=None, place_data=None):
    """Add drop-target markers (always) and debug markers (--markers)."""
    modelTree = ET.parse(OUTPUT_XML)
    root = modelTree.getroot()
    worldbody = root.find("worldbody")

    for i, (px, py) in enumerate(place_positions):
        ET.SubElement(worldbody, "geom", {
            "name": f"drop_target_{i}",
            "type": "cylinder",
            "size": "0.025 0.001",
            "pos": f"{px} {py} {TABLE_HEIGHT + 0.001}",
            "rgba": "1 0.2 0.2 0.8",
            "contype": "0", "conaffinity": "0"
        })

    if show_markers and grasp_data is not None:
        for i, g in enumerate(grasp_data):
            if g is None:
                continue
            for name, key, rgba in [
                (f"target_left_{i}", "target_left", "0 1 0 0.9"),
                (f"target_right_{i}", "target_right", "0 1 0 0.9"),
                (f"grasp_center_{i}", "grasp_center", "1 1 0 0.9"),
                (f"pre_grasp_{i}", "pre_grasp_tip", "0 1 1 0.9"),
            ]:
                if key in g:
                    p = g[key]
                    ET.SubElement(worldbody, "geom", {
                        "name": name, "type": "sphere", "size": "0.008",
                        "pos": f"{p[0]} {p[1]} {p[2]}",
                        "rgba": rgba,
                        "contype": "0", "conaffinity": "0"
                    })

    if show_markers and place_data is not None:
        for i, p in enumerate(place_data):
            if p is None:
                continue
            for name, key, rgba in [
                (f"place_center_{i}", "place_center", "1 0 0 0.9"),
                (f"pre_place_{i}", "pre_place_tip", "1 0.5 0 0.9"),
            ]:
                if key in p:
                    pt = p[key]
                    ET.SubElement(worldbody, "geom", {
                        "name": name, "type": "sphere", "size": "0.008",
                        "pos": f"{pt[0]} {pt[1]} {pt[2]}",
                        "rgba": rgba,
                        "contype": "0", "conaffinity": "0"
                    })

    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


# ══════════════════════════════════════════════════════════════
#  Grasp geometry
# ══════════════════════════════════════════════════════════════

def calculate_grasp_from_pose(piece_pos, piece_quat):
    R_piece = quat_to_rotmat(piece_quat)
    local_base_center = np.array([0.016, 0.016, 0.016])
    grasp_center = piece_pos + R_piece @ local_base_center

    finger_axis = R_piece @ np.array([0.0, 1.0, 0.0])
    finger_axis = finger_axis / np.linalg.norm(finger_axis)

    target_left = grasp_center + GRIPPER_OPEN * finger_axis
    target_right = grasp_center - GRIPPER_OPEN * finger_axis

    gripper_z = np.array([0, 0, -1])
    gripper_y = -finger_axis
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x = gripper_x / np.linalg.norm(gripper_x)
    gripper_y = np.cross(gripper_z, gripper_x)

    R_gripper = np.column_stack([gripper_x, gripper_y, gripper_z])
    gripper_quat = rotmat_to_quat(R_gripper)

    pre_grasp_tip = grasp_center + np.array([0, 0, PRE_GRASP_HEIGHT])
    grasp_tip = grasp_center.copy()
    lift_tip = grasp_center + np.array([0, 0, LIFT_HEIGHT])

    return {
        "grasp_center": grasp_center,
        "finger_axis": finger_axis,
        "target_left": target_left,
        "target_right": target_right,
        "gripper_quat": gripper_quat,
        "pre_grasp_tip": pre_grasp_tip,
        "grasp_tip": grasp_tip,
        "lift_tip": lift_tip,
    }


def calculate_placement_geometry(place_x, place_y, settled_quat, grasp_gripper_quat):
    R_piece = quat_to_rotmat(settled_quat)
    R_gripper_grasp = quat_to_rotmat(grasp_gripper_quat)

    R_gripper_place = R_piece.T @ R_gripper_grasp
    place_quat = rotmat_to_quat(R_gripper_place)

    gripper_z = R_gripper_place[:, 2]

    place_tip = np.array([place_x, place_y, PLACE_TIP_HEIGHT])
    pre_place_tip = place_tip - PRE_PLACE_OFFSET * gripper_z + np.array([0, 0, 0.04])
    retreat_tip = place_tip - PRE_PLACE_OFFSET * gripper_z + np.array([0, 0, 0.06])

    return {
        "place_center": place_tip,
        "place_quat": place_quat,
        "place_R": R_gripper_place,
        "pre_place_tip": pre_place_tip,
        "place_tip": place_tip,
        "retreat_tip": retreat_tip,
    }


# ══════════════════════════════════════════════════════════════
#  IK solver
# ══════════════════════════════════════════════════════════════

def compute_ik(model, data, target_pos, target_quat, q_init, min_elbow_z=None):
    hand_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    nv = 7

    if min_elbow_z is not None:
        elbow_body_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
            for n in ("link3", "link4", "link5")
        ]

    d = mj.MjData(model)
    d.qpos[:7] = q_init.copy()
    d.qpos[7] = GRIPPER_OPEN
    d.qpos[8] = GRIPPER_OPEN
    mj.mj_forward(model, d)

    for i in range(IK_MAX_ITER):
        hand_pos = d.xpos[hand_body_id].copy()
        hand_quat = d.xquat[hand_body_id].copy()
        hand_rot = d.xmat[hand_body_id].reshape(3, 3)
        hand_z = hand_rot[:, 2]
        tip_pos = hand_pos + GRASP_OFFSET * hand_z

        pos_err = target_pos - tip_pos
        ori_err = quat_error(hand_quat, target_quat)
        err = np.concatenate([pos_err, ori_err])
        err_norm = np.linalg.norm(err[:3])

        if err_norm < IK_TOL and np.linalg.norm(err[3:]) < IK_TOL * 5:
            return d.qpos[:7].copy(), True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, d, jacp, jacr, hand_body_id)

        Jp = jacp[:, :nv]
        Jr = jacr[:, :nv]

        offset_world = GRASP_OFFSET * hand_z
        skew = np.array([
            [0, -offset_world[2], offset_world[1]],
            [offset_world[2], 0, -offset_world[0]],
            [-offset_world[1], offset_world[0], 0]
        ])
        Jp_tip = Jp - skew @ Jr
        J = np.vstack([Jp_tip, Jr])

        JJT = J @ J.T + IK_DAMPING * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        if min_elbow_z is not None:
            J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))
            N = np.eye(nv) - J_pinv @ J
            grad = np.zeros(nv)
            for eid in elbow_body_ids:
                z = d.xpos[eid][2]
                if z < min_elbow_z:
                    jac_e = np.zeros((3, model.nv))
                    mj.mj_jacBody(model, d, jac_e, None, eid)
                    grad += jac_e[2, :nv] * (min_elbow_z - z)
            dq += N @ grad * 2.0

        d.qpos[:nv] += IK_STEP * dq

        for j in range(nv):
            lo = model.jnt_range[j, 0]
            hi = model.jnt_range[j, 1]
            if lo < hi:
                d.qpos[j] = np.clip(d.qpos[j], lo, hi)

        mj.mj_forward(model, d)

    return d.qpos[:7].copy(), False


# ══════════════════════════════════════════════════════════════
#  Motion helpers
# ══════════════════════════════════════════════════════════════

def interpolate_joints(q_start, q_end, num_steps):
    traj = np.zeros((num_steps, 7))
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        t = 3 * t**2 - 2 * t**3
        traj[i] = q_start + t * (q_end - q_start)
    return traj


def execute_trajectory(model, data, v, traj, gripper_val, steps_per_wp=50):
    for wp in traj:
        for _ in range(steps_per_wp):
            q = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            data.ctrl[:7] = KP * (wp - q) + KD * (0 - qd) + data.qfrc_bias[:7]
            data.ctrl[7] = gripper_val
            mj.mj_step(model, data)
            if v is not None:
                v.sync()


def hold_position(model, data, v, q_target, gripper_val, duration):
    dt = model.opt.timestep
    steps = int(duration / dt)
    for _ in range(steps):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        data.ctrl[:7] = KP * (q_target - q) + KD * (0 - qd) + data.qfrc_bias[:7]
        data.ctrl[7] = gripper_val
        mj.mj_step(model, data)
        if v is not None:
            v.sync()


# ══════════════════════════════════════════════════════════════
#  RRT Motion Planner
# ══════════════════════════════════════════════════════════════

def plan_motion(model, q_start, q_goal, gripper_val, all_piece_qpos,
                max_rrt_iter=5000, step_size=0.15, goal_bias=0.2,
                smooth_iter=200):
    """
    Plan a collision-free joint-space trajectory using RRT + shortcut smoothing.
    Checks robot vs all tables + ground contacts.
    all_piece_qpos: flat array of length NUM_TABLES*7 with all piece freejoints.
    """
    check_data = mj.MjData(model)

    # Obstacle geoms: all tables + ground
    table_ids = []
    for i in range(NUM_TABLES):
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"table_{i}")
        if gid >= 0:
            table_ids.append(gid)
    ground_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
    obstacles = frozenset(set(table_ids) | {ground_id})

    # Skip bodies: all pieces + world + link0
    piece_bodies = set()
    for i in range(NUM_TABLES):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"ChessPiece_{i}")
        if bid >= 0:
            piece_bodies.add(bid)
    link0_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link0")
    skip_bodies = frozenset(piece_bodies | {0, link0_body})

    joint_lo = np.array([model.jnt_range[i, 0] for i in range(7)])
    joint_hi = np.array([model.jnt_range[i, 1] for i in range(7)])

    n_piece_dof = len(all_piece_qpos)

    def col_free(q):
        check_data.qpos[:7] = q
        check_data.qpos[7] = gripper_val
        check_data.qpos[8] = gripper_val
        check_data.qpos[9:9 + n_piece_dof] = all_piece_qpos
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

    # Try direct path
    if edge_free(q_start, q_goal, n=20):
        return interpolate_joints(q_start, q_goal, 60)

    if not col_free(q_start):
        print("    WARNING: start in collision — using direct path")
        return interpolate_joints(q_start, q_goal, 60)
    if not col_free(q_goal):
        print("    WARNING: goal in collision — using direct path")
        return interpolate_joints(q_start, q_goal, 60)

    # RRT
    nodes = [q_start.copy()]
    parents = [-1]
    path = None

    for it in range(max_rrt_iter):
        if np.random.random() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = np.random.uniform(joint_lo, joint_hi)

        dists = np.linalg.norm(np.array(nodes) - q_rand, axis=1)
        nearest_idx = int(np.argmin(dists))
        q_nearest = nodes[nearest_idx]

        diff = q_rand - q_nearest
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            continue
        if dist > step_size:
            q_new = q_nearest + step_size * (diff / dist)
        else:
            q_new = q_rand.copy()
        q_new = np.clip(q_new, joint_lo, joint_hi)

        if not edge_free(q_nearest, q_new):
            continue

        nodes.append(q_new.copy())
        parents.append(nearest_idx)

        if np.linalg.norm(q_new - q_goal) < step_size:
            if edge_free(q_new, q_goal):
                nodes.append(q_goal.copy())
                parents.append(len(nodes) - 2)
                path = []
                idx = len(nodes) - 1
                while idx != -1:
                    path.append(nodes[idx])
                    idx = parents[idx]
                path.reverse()
                break

    if path is None:
        print(f"    RRT FAILED ({len(nodes)} nodes) — falling back to direct path")
        return interpolate_joints(q_start, q_goal, 60)

    # Shortcut smoothing
    for _ in range(smooth_iter):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 1)
        j = np.random.randint(i + 2, len(path))
        if edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]

    # Densify
    segments = []
    for i in range(len(path) - 1):
        d = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
        n_pts = max(10, int(d / 0.02))
        segments.append(interpolate_joints(np.array(path[i]),
                                           np.array(path[i + 1]), n_pts))
    return np.vstack(segments)


# ══════════════════════════════════════════════════════════════
#  Experiment runner
# ══════════════════════════════════════════════════════════════

def run_experiment(seed=None, show_viewer=True, show_markers=False):
    """Run one full 4-piece pick-and-place experiment. Returns list of metrics dicts."""
    if seed is not None:
        np.random.seed(seed)

    # ── Generate random poses for all pieces ──
    piece_poses = []
    for table in TABLES:
        pos, quat, yaw = random_piece_pose(table)
        piece_poses.append((pos, quat))

    # ── Build scene ──
    build_scene(piece_poses)

    # ── Load and settle ──
    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = BASE_HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    mj.mj_forward(model, data)

    print("Settling pieces...")
    for _ in range(3000):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        data.ctrl[:7] = KP * (BASE_HOME_Q - q) + KD * (0 - qd) + data.qfrc_bias[:7]
        data.ctrl[7] = GRIPPER_OPEN
        mj.mj_step(model, data)

    # ── Read settled poses ──
    settled = []
    for i in range(NUM_TABLES):
        idx = 9 + i * 7
        pos = data.qpos[idx:idx + 3].copy()
        quat = data.qpos[idx + 3:idx + 7].copy()
        settled.append((pos, quat))

    # ── Compute grasp + placement geometry for all pieces ──
    grasps = []
    places = []
    place_positions = []
    for i, table in enumerate(TABLES):
        geom = calculate_grasp_from_pose(settled[i][0], settled[i][1])
        grasps.append(geom)

        grasp_xy = geom["grasp_center"][:2]
        px, py = random_place_pos(table, grasp_xy)
        place_positions.append((px, py))

        pg = calculate_placement_geometry(px, py, settled[i][1], geom["gripper_quat"])
        places.append(pg)

    # ── Add markers and reload ──
    add_markers_to_scene(
        place_positions, show_markers=show_markers,
        grasp_data=grasps if show_markers else None,
        place_data=places if show_markers else None,
    )

    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = BASE_HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    for i in range(NUM_TABLES):
        idx = 9 + i * 7
        data.qpos[idx:idx + 3] = settled[i][0]
        data.qpos[idx + 3:idx + 7] = settled[i][1]
    mj.mj_forward(model, data)

    # ── Inflate all tables for planning ──
    table_geom_ids = []
    orig_sizes = []
    for i in range(NUM_TABLES):
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"table_{i}")
        table_geom_ids.append(gid)
        orig_sizes.append(model.geom_size[gid].copy())
        model.geom_size[gid][2] += COLLISION_MARGIN

    min_z = TABLE_HEIGHT + COLLISION_MARGIN + 0.02

    # All piece qpos as flat array for collision checking
    all_piece_qpos = np.concatenate(
        [np.concatenate([s[0], s[1]]) for s in settled]
    )

    # ── Solve IK and plan trajectories for all pieces ──
    piece_plans = []
    prev_retreat_q = None

    for i, table in enumerate(TABLES):
        home_q = home_q_for_table(table)
        geom = grasps[i]
        place = places[i]
        grasp_quat = geom["gripper_quat"]
        place_quat = place["place_quat"]

        print(f"\n--- Piece {i} ({table['name']}) IK ---")
        q_pre_grasp, ok1 = compute_ik(model, data, geom["pre_grasp_tip"], grasp_quat, home_q, min_elbow_z=min_z)
        q_grasp, ok2 = compute_ik(model, data, geom["grasp_tip"], grasp_quat, q_pre_grasp, min_elbow_z=min_z)
        q_lift, ok3 = compute_ik(model, data, geom["lift_tip"], grasp_quat, q_grasp, min_elbow_z=min_z)
        q_pre_place, ok4 = compute_ik(model, data, place["pre_place_tip"], place_quat, q_lift, min_elbow_z=min_z)
        q_place, ok5 = compute_ik(model, data, place["place_tip"], place_quat, q_pre_place, min_elbow_z=min_z)
        q_retreat, ok6 = compute_ik(model, data, place["retreat_tip"], place_quat, q_place, min_elbow_z=min_z)

        ik_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
        if not ik_ok:
            print(f"  WARNING: Not all IK converged for piece {i}")

        # Plan trajectories
        print(f"--- Piece {i} ({table['name']}) Planning ---")
        start_q = prev_retreat_q if prev_retreat_q is not None else BASE_HOME_Q

        trajs = {}
        print(f"  → HOME_{i}:")
        trajs["to_home"] = plan_motion(model, start_q, home_q, GRIPPER_OPEN, all_piece_qpos)
        print(f"  HOME_{i} → PRE-GRASP:")
        trajs["to_pre"] = plan_motion(model, home_q, q_pre_grasp, GRIPPER_OPEN, all_piece_qpos)
        print(f"  PRE-GRASP → GRASP:")
        trajs["to_grasp"] = plan_motion(model, q_pre_grasp, q_grasp, GRIPPER_OPEN, all_piece_qpos)
        print(f"  GRASP → LIFT:")
        trajs["to_lift"] = plan_motion(model, q_grasp, q_lift, GRIPPER_CLOSED, all_piece_qpos)
        print(f"  LIFT → PRE-PLACE:")
        trajs["to_pre_place"] = plan_motion(model, q_lift, q_pre_place, GRIPPER_CLOSED, all_piece_qpos)
        print(f"  PRE-PLACE → PLACE:")
        trajs["to_place"] = plan_motion(model, q_pre_place, q_place, GRIPPER_CLOSED, all_piece_qpos)
        print(f"  PLACE → RETREAT:")
        trajs["to_retreat"] = plan_motion(model, q_place, q_retreat, GRIPPER_OPEN, all_piece_qpos)

        piece_plans.append({
            "home_q": home_q,
            "q_pre_grasp": q_pre_grasp,
            "q_grasp": q_grasp,
            "q_lift": q_lift,
            "q_pre_place": q_pre_place,
            "q_place": q_place,
            "q_retreat": q_retreat,
            "trajs": trajs,
            "ik_ok": ik_ok,
        })
        prev_retreat_q = q_retreat

    # Restore table sizes for simulation
    for gid, orig in zip(table_geom_ids, orig_sizes):
        model.geom_size[gid] = orig

    # ── Launch viewer (if not eval mode) ──
    v = None
    if show_viewer:
        v = viewer.launch_passive(model, data)
        v.cam.distance = 1.5
        v.cam.azimuth = 135
        v.cam.elevation = -30
        v.cam.lookat[:] = [0, 0, TABLE_HEIGHT + 0.1]

    # ── Execute all pick-and-place sequences ──
    metrics = []

    try:
        hold_position(model, data, v, BASE_HOME_Q, GRIPPER_OPEN, PAUSE_SECONDS)

        for i, plan in enumerate(piece_plans):
            table = TABLES[i]
            print(f"\n>>> Piece {i} ({table['name']})")

            # Move to this table's home
            print(f"  Moving to HOME_{i}...")
            execute_trajectory(model, data, v, plan["trajs"]["to_home"], GRIPPER_OPEN)
            hold_position(model, data, v, plan["home_q"], GRIPPER_OPEN, PAUSE_SECONDS)

            # PRE-GRASP
            print(f"  Moving to PRE-GRASP...")
            execute_trajectory(model, data, v, plan["trajs"]["to_pre"], GRIPPER_OPEN)
            hold_position(model, data, v, plan["q_pre_grasp"], GRIPPER_OPEN, PAUSE_SECONDS)

            # GRASP
            print(f"  Lowering to GRASP...")
            execute_trajectory(model, data, v, plan["trajs"]["to_grasp"], GRIPPER_OPEN)
            hold_position(model, data, v, plan["q_grasp"], GRIPPER_OPEN, PAUSE_SECONDS)

            # CLOSE
            print(f"  Closing gripper...")
            hold_position(model, data, v, plan["q_grasp"], GRIPPER_CLOSED, 2.0)

            # LIFT
            print(f"  Lifting...")
            execute_trajectory(model, data, v, plan["trajs"]["to_lift"], GRIPPER_CLOSED)
            hold_position(model, data, v, plan["q_lift"], GRIPPER_CLOSED, PAUSE_SECONDS)

            # PRE-PLACE
            print(f"  Moving to PRE-PLACE...")
            execute_trajectory(model, data, v, plan["trajs"]["to_pre_place"], GRIPPER_CLOSED)
            hold_position(model, data, v, plan["q_pre_place"], GRIPPER_CLOSED, PAUSE_SECONDS)

            # PLACE
            print(f"  Lowering to PLACE...")
            execute_trajectory(model, data, v, plan["trajs"]["to_place"], GRIPPER_CLOSED)
            hold_position(model, data, v, plan["q_place"], GRIPPER_CLOSED, PAUSE_SECONDS)

            # OPEN
            print(f"  Releasing piece...")
            hold_position(model, data, v, plan["q_place"], GRIPPER_OPEN, 2.0)

            # RETREAT
            print(f"  Retreating...")
            execute_trajectory(model, data, v, plan["trajs"]["to_retreat"], GRIPPER_OPEN)
            hold_position(model, data, v, plan["q_retreat"], GRIPPER_OPEN, PAUSE_SECONDS)

            # Let piece settle after release
            hold_position(model, data, v, plan["q_retreat"], GRIPPER_OPEN, 1.0)

            # ── Measure metrics ──
            idx = 9 + i * 7
            final_pos = data.qpos[idx:idx + 3].copy()
            final_quat = data.qpos[idx + 3:idx + 7].copy()
            R_final = quat_to_rotmat(final_quat)

            target_xy = np.array(place_positions[i])
            actual_xy = final_pos[:2]
            xy_error = np.linalg.norm(target_xy - actual_xy)
            upright = R_final[2, 2] > 0.95

            metrics.append({
                "table": table["name"],
                "piece_idx": i,
                "target_xy": target_xy,
                "actual_xy": actual_xy,
                "xy_error": xy_error,
                "upright": upright,
                "ik_ok": plan["ik_ok"],
            })

            print(f"  XY error: {xy_error:.4f}m | Upright: {upright}")

        print("\n>>> ALL PIECES DONE.")

        if v is not None:
            print("Close viewer to exit.")
            while v.is_running():
                q_last = piece_plans[-1]["q_retreat"]
                hold_position(model, data, v, q_last, GRIPPER_OPEN, 0.05)

    except KeyboardInterrupt:
        pass

    if v is not None:
        v.close()

    return metrics


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def print_metrics_table(all_metrics):
    """Print a formatted results table from a list of experiment metrics."""
    # Header
    print("\n" + "=" * 80)
    print(f"{'Seed':>6} | {'Table':<10} | {'XY Err (mm)':>11} | {'Upright':>7} | {'IK OK':>5}")
    print("-" * 80)

    all_xy = []
    all_upright = []
    all_ik = []

    for seed, experiment_metrics in all_metrics:
        for m in experiment_metrics:
            xy_mm = m["xy_error"] * 1000
            up_str = "Y" if m["upright"] else "N"
            ik_str = "Y" if m["ik_ok"] else "N"
            print(f"{seed:>6} | {m['table']:<10} | {xy_mm:>11.1f} | {up_str:>7} | {ik_str:>5}")
            all_xy.append(m["xy_error"])
            all_upright.append(m["upright"])
            all_ik.append(m["ik_ok"])

    # Aggregate stats
    print("=" * 80)
    n = len(all_xy)
    if n > 0:
        mean_xy = np.mean(all_xy) * 1000
        std_xy = np.std(all_xy) * 1000
        max_xy = np.max(all_xy) * 1000
        upright_pct = sum(all_upright) / n * 100
        ik_pct = sum(all_ik) / n * 100
        print(f"  Samples:       {n}")
        print(f"  XY error:      {mean_xy:.1f} +/- {std_xy:.1f} mm  (max {max_xy:.1f} mm)")
        print(f"  Upright rate:  {upright_pct:.1f}%")
        print(f"  IK success:    {ik_pct:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IK grasp & place — 4 tables")
    parser.add_argument("seed", nargs="?", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--markers", action="store_true",
                        help="Show all debug markers (grasp targets, etc.)")
    parser.add_argument("--eval", type=int, default=None, metavar="N",
                        help="Run N headless experiments and print results table")
    args = parser.parse_args()

    if args.eval is not None:
        print(f"Running {args.eval} evaluation experiments (headless)...")
        all_metrics = []
        for exp in range(args.eval):
            seed = exp if args.seed is None else args.seed + exp
            print(f"\n{'='*60}")
            print(f"  EXPERIMENT {exp + 1}/{args.eval}  (seed={seed})")
            print(f"{'='*60}")
            m = run_experiment(seed=seed, show_viewer=False, show_markers=False)
            all_metrics.append((seed, m))
        print_metrics_table(all_metrics)
    else:
        seed = args.seed
        if seed is not None:
            print(f"Using seed: {seed}")
        print("=" * 60)
        print("  IK GRASP & PLACE — 4 TABLES")
        print("=" * 60)
        metrics = run_experiment(seed=seed, show_viewer=True, show_markers=args.markers)

        print("\n--- Results ---")
        for m in metrics:
            xy_mm = m["xy_error"] * 1000
            print(f"  {m['table']:<10}: XY err = {xy_mm:.1f}mm, Upright = {m['upright']}")
