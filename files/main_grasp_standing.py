"""
IK-based top-down grasp for a chess piece standing upright on a table.

Single robot, single piece on the front table.
The piece spawns standing (no 90° lay-down rotation).
Gripper approaches from above and grasps the stem.

Markers (always shown, non-colliding):
  CYAN    sphere = Pre-grasp waypoint
  YELLOW  sphere = Grasp center (stem mid-point)
  GREEN   spheres = Left / right finger targets
  MAGENTA sphere = Lift waypoint
  ORANGE  sphere = Pre-place waypoint
  RED     cylinder = Drop target on table
  RED     sphere = Place waypoint
  WHITE   sphere = Retreat waypoint
  BLUE    small spheres = Descent path (pre-grasp → grasp)
  PINK    small spheres = Approach path (home → pre-grasp)
"""
import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import argparse
import sys

# ── Scene config ──────────────────────────────────────────────
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
OUTPUT_XML = "franka_emika_panda/grasp_ik_scene_standing.xml"

BASE_HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float)

TABLE_HEIGHT = 0.3
TABLE_HALF = 0.15
COLLISION_MARGIN = 0.05

# Grasp geometry
GRASP_OFFSET = 0.103
PRE_GRASP_HEIGHT = 0.10
LIFT_HEIGHT = 0.15

# Placement
PLACE_MIN_DIST = 0.08
PLACE_TIP_HEIGHT = TABLE_HEIGHT + 0.06
PRE_PLACE_OFFSET = 0.10

# Timing
PAUSE_SECONDS = 0.5

# IK parameters
IK_TOL = 1e-3
IK_MAX_ITER = 200
IK_STEP = 0.3
IK_DAMPING = 1e-4

# Spawn config — standing piece
SPAWN_OFFSET = 0.06
SPAWN_Z = TABLE_HEIGHT + 0.002
SPAWN_YAW_RANGE = (-0.5, 0.5)
PLACE_OFFSET = 0.08

NUM_PIECES = 1

# Single table — front
TABLE = {"name": "front", "cx": 0.45, "cy": 0.00}

# Piece local geometry (from collision model in XML)
# Base: box half-size 0.016 at local (0.016, 0.016, 0.016) → 32mm cube, bottom at z=0
# Stem: cylinder r=0.010, half-h=0.030 at local (0.016, 0.016, 0.060) → z=[0.030, 0.090]
STEM_LOCAL_CENTER = np.array([0.016, 0.016, 0.060])
STEM_RADIUS = 0.010
STEM_HALF_HEIGHT = 0.030
BASE_TOP_Z_LOCAL = 0.032
PIECE_TOP_Z_LOCAL = 0.090


def home_q_for_table(table):
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

def standing_piece_pose(table):
    """Generate random (pos, quat, yaw) for piece standing upright on the table."""
    cx, cy = table["cx"], table["cy"]
    x = np.random.uniform(cx - SPAWN_OFFSET, cx + SPAWN_OFFSET)
    y = np.random.uniform(cy - SPAWN_OFFSET, cy + SPAWN_OFFSET)
    yaw = np.random.uniform(*SPAWN_YAW_RANGE)

    q_yaw = axis_angle_to_quat(np.array([0, 0, 1]), yaw)
    return [x, y, SPAWN_Z], q_yaw, yaw


def random_place_pos(table, grasp_xy):
    cx, cy = table["cx"], table["cy"]
    for _ in range(100):
        x = np.random.uniform(cx - PLACE_OFFSET, cx + PLACE_OFFSET)
        y = np.random.uniform(cy - PLACE_OFFSET, cy + PLACE_OFFSET)
        dist = np.linalg.norm(np.array([x, y]) - grasp_xy)
        if dist >= PLACE_MIN_DIST:
            return x, y
    return cx, cy


def build_scene(piece_pos, piece_quat):
    """Build XML scene with 1 table and 1 standing chess piece."""
    modelTree = ET.parse(ROOT_MODEL_XML)
    root = modelTree.getroot()

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {
        "name": "chess_king_mesh",
        "file": "chess_king.stl",
        "scale": "0.001 0.001 0.001"
    })

    worldbody = root.find("worldbody")

    # ── Table ──
    table_half_h = TABLE_HEIGHT / 2.0
    ET.SubElement(worldbody, "geom", {
        "name": "table_0",
        "type": "box",
        "size": f"{TABLE_HALF} {TABLE_HALF} {table_half_h}",
        "pos": f"{TABLE['cx']} {TABLE['cy']} {table_half_h}",
        "rgba": "0.4 0.3 0.2 1",
        "friction": "1.5 0.5 0.1",
        "contype": "1", "conaffinity": "1"
    })

    # ── Chess piece (standing) ──
    body = ET.SubElement(worldbody, "body", {
        "name": "ChessPiece_0",
        "pos": f"{piece_pos[0]} {piece_pos[1]} {piece_pos[2]}",
        "quat": f"{piece_quat[0]} {piece_quat[1]} {piece_quat[2]} {piece_quat[3]}"
    })
    ET.SubElement(body, "freejoint", {"name": "chess_piece_joint_0"})

    ET.SubElement(body, "geom", {
        "name": "chess_piece_visual_0",
        "type": "mesh", "mesh": "chess_king_mesh",
        "rgba": "0.85 0.75 0.55 1",
        "contype": "0", "conaffinity": "0", "mass": "0"
    })
    ET.SubElement(body, "geom", {
        "name": "chess_piece_base_col_0",
        "type": "box",
        "size": "0.016 0.016 0.016",
        "pos": "0.016 0.016 0.016",
        "rgba": "0.85 0.75 0.55 0.3",
        "density": "500",
        "friction": "1.5 1.5 0.5",
        "contype": "1", "conaffinity": "1", "condim": "4"
    })
    ET.SubElement(body, "geom", {
        "name": "chess_piece_stem_col_0",
        "type": "cylinder",
        "size": "0.010 0.030",
        "pos": "0.016 0.016 0.060",
        "rgba": "0.85 0.75 0.55 0.2",
        "density": "500",
        "friction": "1.5 0.5 0.1",
        "contype": "1", "conaffinity": "1", "condim": "4"
    })

    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


def add_markers_to_scene(grasp_data, place_data, place_pos):
    """Add plenty of non-colliding markers to visualize all waypoints."""
    modelTree = ET.parse(OUTPUT_XML)
    root = modelTree.getroot()
    worldbody = root.find("worldbody")
    marker_idx = 0

    def add_sphere(name, pos, rgba, size="0.008"):
        ET.SubElement(worldbody, "geom", {
            "name": name, "type": "sphere", "size": size,
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "rgba": rgba,
            "contype": "0", "conaffinity": "0"
        })

    def add_cylinder(name, pos, rgba, size="0.025 0.001"):
        ET.SubElement(worldbody, "geom", {
            "name": name, "type": "cylinder", "size": size,
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "rgba": rgba,
            "contype": "0", "conaffinity": "0"
        })

    # ── Grasp markers ──
    g = grasp_data

    # Finger targets (green)
    add_sphere("marker_finger_left", g["target_left"], "0 1 0 0.9")
    add_sphere("marker_finger_right", g["target_right"], "0 1 0 0.9")

    # Grasp center (yellow)
    add_sphere("marker_grasp_center", g["grasp_center"], "1 1 0 0.9", "0.010")

    # Pre-grasp (cyan)
    add_sphere("marker_pre_grasp", g["pre_grasp_tip"], "0 1 1 0.9", "0.010")

    # Lift (magenta)
    add_sphere("marker_lift", g["lift_tip"], "1 0 1 0.9", "0.010")

    # Descent path: pre-grasp → grasp (blue small spheres)
    n_descent = 8
    for k in range(1, n_descent):
        t = k / n_descent
        p = g["pre_grasp_tip"] + t * (g["grasp_tip"] - g["pre_grasp_tip"])
        add_sphere(f"marker_descent_{k}", p, "0.3 0.3 1 0.7", "0.004")

    # Ascent path: grasp → lift (magenta small spheres)
    n_ascent = 6
    for k in range(1, n_ascent):
        t = k / n_ascent
        p = g["grasp_tip"] + t * (g["lift_tip"] - g["grasp_tip"])
        add_sphere(f"marker_ascent_{k}", p, "1 0 1 0.5", "0.004")

    # Finger axis visualization: small markers along finger approach
    for k in range(1, 5):
        t = k / 5.0
        p_left = g["grasp_center"] + t * GRIPPER_OPEN * g["finger_axis"]
        p_right = g["grasp_center"] - t * GRIPPER_OPEN * g["finger_axis"]
        add_sphere(f"marker_faxis_l_{k}", p_left, "0 0.7 0 0.5", "0.003")
        add_sphere(f"marker_faxis_r_{k}", p_right, "0 0.7 0 0.5", "0.003")

    # ── Place markers ──
    px, py = place_pos
    p = place_data

    # Drop target disc on table (red cylinder)
    add_cylinder("marker_drop_target", [px, py, TABLE_HEIGHT + 0.001], "1 0.2 0.2 0.8")

    # Pre-place (orange)
    add_sphere("marker_pre_place", p["pre_place_tip"], "1 0.5 0 0.9", "0.010")

    # Place (red sphere)
    add_sphere("marker_place", p["place_tip"], "1 0 0 0.9", "0.010")

    # Retreat (white)
    add_sphere("marker_retreat", p["retreat_tip"], "1 1 1 0.9", "0.010")

    # Transfer path: lift → pre-place (pink small spheres)
    n_transfer = 8
    for k in range(1, n_transfer):
        t = k / n_transfer
        pt = g["lift_tip"] + t * (p["pre_place_tip"] - g["lift_tip"])
        add_sphere(f"marker_transfer_{k}", pt, "1 0.6 0.8 0.5", "0.004")

    # Place descent path: pre-place → place (orange small spheres)
    n_place_desc = 6
    for k in range(1, n_place_desc):
        t = k / n_place_desc
        pt = p["pre_place_tip"] + t * (p["place_tip"] - p["pre_place_tip"])
        add_sphere(f"marker_place_desc_{k}", pt, "1 0.5 0 0.5", "0.004")

    # Gripper orientation indicator at grasp: short line along gripper Z (down)
    gz_start = g["grasp_center"] + np.array([0, 0, 0.02])
    gz_end = g["grasp_center"] - np.array([0, 0, 0.02])
    for k in range(5):
        t = k / 4.0
        pt = gz_start + t * (gz_end - gz_start)
        add_sphere(f"marker_gz_{k}", pt, "0.5 0.5 0.5 0.6", "0.002")

    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


# ══════════════════════════════════════════════════════════════
#  Grasp geometry — top-down on standing piece stem
# ══════════════════════════════════════════════════════════════

def calculate_grasp_standing(piece_pos, piece_quat):
    """Compute top-down grasp geometry for a standing piece.

    Targets the stem center. Gripper Z points straight down.
    Finger axis is the piece's local Y axis projected to horizontal.
    """
    R_piece = quat_to_rotmat(piece_quat)

    grasp_center = piece_pos + R_piece @ STEM_LOCAL_CENTER

    # Finger approach axis: piece local Y projected to horizontal plane
    finger_axis = R_piece @ np.array([0.0, 1.0, 0.0])
    finger_axis[2] = 0.0
    finger_axis = finger_axis / np.linalg.norm(finger_axis)

    target_left = grasp_center + GRIPPER_OPEN * finger_axis
    target_right = grasp_center - GRIPPER_OPEN * finger_axis

    # Top-down gripper orientation
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


def calculate_placement_geometry(place_x, place_y, grasp_gripper_quat):
    """Placement keeps the piece upright — just move it to target location."""
    gripper_quat = grasp_gripper_quat
    R_gripper = quat_to_rotmat(gripper_quat)
    gripper_z = R_gripper[:, 2]

    place_tip = np.array([place_x, place_y, PLACE_TIP_HEIGHT])
    pre_place_tip = place_tip - PRE_PLACE_OFFSET * gripper_z + np.array([0, 0, 0.04])
    retreat_tip = place_tip - PRE_PLACE_OFFSET * gripper_z + np.array([0, 0, 0.06])

    return {
        "place_center": place_tip,
        "place_quat": gripper_quat,
        "place_R": R_gripper,
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

def plan_motion(model, q_start, q_goal, gripper_val, piece_qpos,
                max_rrt_iter=5000, step_size=0.15, goal_bias=0.2,
                smooth_iter=200):
    check_data = mj.MjData(model)

    table_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_0")
    ground_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
    obstacles = frozenset({table_id, ground_id})

    piece_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ChessPiece_0")
    link0_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link0")
    skip_bodies = frozenset({piece_body, 0, link0_body})

    joint_lo = np.array([model.jnt_range[i, 0] for i in range(7)])
    joint_hi = np.array([model.jnt_range[i, 1] for i in range(7)])

    n_piece_dof = len(piece_qpos)

    def col_free(q):
        check_data.qpos[:7] = q
        check_data.qpos[7] = gripper_val
        check_data.qpos[8] = gripper_val
        check_data.qpos[9:9 + n_piece_dof] = piece_qpos
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
        return interpolate_joints(q_start, q_goal, 60)

    if not col_free(q_start):
        print("    WARNING: start in collision — using direct path")
        return interpolate_joints(q_start, q_goal, 60)
    if not col_free(q_goal):
        print("    WARNING: goal in collision — using direct path")
        return interpolate_joints(q_start, q_goal, 60)

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

    for _ in range(smooth_iter):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 1)
        j = np.random.randint(i + 2, len(path))
        if edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]

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

def run_experiment(seed=None, show_viewer=True):
    if seed is not None:
        np.random.seed(seed)

    # ── Generate standing piece pose ──
    pos, quat, yaw = standing_piece_pose(TABLE)
    print(f"Piece spawned at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), yaw={yaw:.2f} rad")

    # ── Build scene ──
    build_scene(pos, quat)

    # ── Load and settle ──
    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = BASE_HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    mj.mj_forward(model, data)

    print("Settling piece...")
    for _ in range(3000):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        data.ctrl[:7] = KP * (BASE_HOME_Q - q) + KD * (0 - qd) + data.qfrc_bias[:7]
        data.ctrl[7] = GRIPPER_OPEN
        mj.mj_step(model, data)

    # ── Read settled pose ──
    idx = 9
    settled_pos = data.qpos[idx:idx + 3].copy()
    settled_quat = data.qpos[idx + 3:idx + 7].copy()
    print(f"Settled at ({settled_pos[0]:.3f}, {settled_pos[1]:.3f}, {settled_pos[2]:.3f})")

    # ── Compute grasp + placement ──
    grasp = calculate_grasp_standing(settled_pos, settled_quat)

    grasp_xy = grasp["grasp_center"][:2]
    px, py = random_place_pos(TABLE, grasp_xy)
    place = calculate_placement_geometry(px, py, grasp["gripper_quat"])

    print(f"Grasp center: ({grasp['grasp_center'][0]:.3f}, {grasp['grasp_center'][1]:.3f}, {grasp['grasp_center'][2]:.3f})")
    print(f"Place target: ({px:.3f}, {py:.3f})")

    # ── Add markers and reload ──
    add_markers_to_scene(grasp, place, (px, py))

    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = BASE_HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    data.qpos[idx:idx + 3] = settled_pos
    data.qpos[idx + 3:idx + 7] = settled_quat
    mj.mj_forward(model, data)

    # ── Inflate table for planning ──
    table_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_0")
    orig_size = model.geom_size[table_gid].copy()
    model.geom_size[table_gid][2] += COLLISION_MARGIN

    min_z = TABLE_HEIGHT + COLLISION_MARGIN + 0.02

    piece_qpos = np.concatenate([settled_pos, settled_quat])

    # ── Solve IK ──
    home_q = home_q_for_table(TABLE)
    grasp_quat = grasp["gripper_quat"]
    place_quat = place["place_quat"]

    print("\n--- IK ---")
    q_pre_grasp, ok1 = compute_ik(model, data, grasp["pre_grasp_tip"], grasp_quat, home_q, min_elbow_z=min_z)
    q_grasp, ok2     = compute_ik(model, data, grasp["grasp_tip"], grasp_quat, q_pre_grasp, min_elbow_z=min_z)
    q_lift, ok3       = compute_ik(model, data, grasp["lift_tip"], grasp_quat, q_grasp, min_elbow_z=min_z)
    q_pre_place, ok4  = compute_ik(model, data, place["pre_place_tip"], place_quat, q_lift, min_elbow_z=min_z)
    q_place, ok5      = compute_ik(model, data, place["place_tip"], place_quat, q_pre_place, min_elbow_z=min_z)
    q_retreat, ok6    = compute_ik(model, data, place["retreat_tip"], place_quat, q_place, min_elbow_z=min_z)

    ik_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
    print(f"  pre_grasp: {'OK' if ok1 else 'FAIL'}  grasp: {'OK' if ok2 else 'FAIL'}  "
          f"lift: {'OK' if ok3 else 'FAIL'}  pre_place: {'OK' if ok4 else 'FAIL'}  "
          f"place: {'OK' if ok5 else 'FAIL'}  retreat: {'OK' if ok6 else 'FAIL'}")

    # ── Plan motions ──
    print("\n--- Planning ---")
    print("  HOME → PRE-GRASP:")
    traj_to_pre = plan_motion(model, home_q, q_pre_grasp, GRIPPER_OPEN, piece_qpos)
    print("  PRE-GRASP → GRASP:")
    traj_to_grasp = plan_motion(model, q_pre_grasp, q_grasp, GRIPPER_OPEN, piece_qpos)
    print("  GRASP → LIFT:")
    traj_to_lift = plan_motion(model, q_grasp, q_lift, GRIPPER_CLOSED, piece_qpos)
    print("  LIFT → PRE-PLACE:")
    traj_to_pre_place = plan_motion(model, q_lift, q_pre_place, GRIPPER_CLOSED, piece_qpos)
    print("  PRE-PLACE → PLACE:")
    traj_to_place = plan_motion(model, q_pre_place, q_place, GRIPPER_CLOSED, piece_qpos)
    print("  PLACE → RETREAT:")
    traj_to_retreat = plan_motion(model, q_place, q_retreat, GRIPPER_OPEN, piece_qpos)

    # Restore table size
    model.geom_size[table_gid] = orig_size

    # ── Launch viewer ──
    v = None
    if show_viewer:
        v = viewer.launch_passive(model, data)
        v.cam.distance = 1.2
        v.cam.azimuth = 135
        v.cam.elevation = -30
        v.cam.lookat[:] = [TABLE["cx"] * 0.5, TABLE["cy"] * 0.5, TABLE_HEIGHT + 0.1]

    # ── Execute pick-and-place ──
    try:
        hold_position(model, data, v, BASE_HOME_Q, GRIPPER_OPEN, PAUSE_SECONDS)

        print("\n>>> Moving to HOME...")
        traj_home = interpolate_joints(BASE_HOME_Q, home_q, 60)
        execute_trajectory(model, data, v, traj_home, GRIPPER_OPEN)
        hold_position(model, data, v, home_q, GRIPPER_OPEN, PAUSE_SECONDS)

        print(">>> Moving to PRE-GRASP...")
        execute_trajectory(model, data, v, traj_to_pre, GRIPPER_OPEN)
        hold_position(model, data, v, q_pre_grasp, GRIPPER_OPEN, PAUSE_SECONDS)

        print(">>> Lowering to GRASP...")
        execute_trajectory(model, data, v, traj_to_grasp, GRIPPER_OPEN)
        hold_position(model, data, v, q_grasp, GRIPPER_OPEN, PAUSE_SECONDS)

        print(">>> Closing gripper...")
        hold_position(model, data, v, q_grasp, GRIPPER_CLOSED, 2.0)

        print(">>> Lifting...")
        execute_trajectory(model, data, v, traj_to_lift, GRIPPER_CLOSED)
        hold_position(model, data, v, q_lift, GRIPPER_CLOSED, PAUSE_SECONDS)

        print(">>> Moving to PRE-PLACE...")
        execute_trajectory(model, data, v, traj_to_pre_place, GRIPPER_CLOSED)
        hold_position(model, data, v, q_pre_place, GRIPPER_CLOSED, PAUSE_SECONDS)

        print(">>> Lowering to PLACE...")
        execute_trajectory(model, data, v, traj_to_place, GRIPPER_CLOSED)
        hold_position(model, data, v, q_place, GRIPPER_CLOSED, PAUSE_SECONDS)

        print(">>> Releasing piece...")
        hold_position(model, data, v, q_place, GRIPPER_OPEN, 2.0)

        print(">>> Retreating...")
        execute_trajectory(model, data, v, traj_to_retreat, GRIPPER_OPEN)
        hold_position(model, data, v, q_retreat, GRIPPER_OPEN, PAUSE_SECONDS)

        # Let piece settle
        hold_position(model, data, v, q_retreat, GRIPPER_OPEN, 1.0)

        # ── Measure ──
        final_pos = data.qpos[idx:idx + 3].copy()
        final_quat = data.qpos[idx + 3:idx + 7].copy()
        R_final = quat_to_rotmat(final_quat)

        target_xy = np.array([px, py])
        actual_xy = final_pos[:2]
        xy_error = np.linalg.norm(target_xy - actual_xy)
        upright = R_final[2, 2] > 0.95

        print(f"\n--- Results ---")
        print(f"  XY error: {xy_error * 1000:.1f} mm")
        print(f"  Upright:  {upright}")
        print(f"  IK OK:    {ik_ok}")

        print("\n>>> DONE. Close viewer to exit.")
        if v is not None:
            while v.is_running():
                hold_position(model, data, v, q_retreat, GRIPPER_OPEN, 0.05)

    except KeyboardInterrupt:
        pass

    if v is not None:
        v.close()

    return {
        "xy_error": xy_error,
        "upright": upright,
        "ik_ok": ik_ok,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-down grasp of standing chess piece — single table")
    parser.add_argument("seed", nargs="?", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        print(f"Using seed: {seed}")
    print("=" * 60)
    print("  TOP-DOWN GRASP — STANDING PIECE")
    print("=" * 60)
    run_experiment(seed=seed, show_viewer=True)
