"""
IK-based grasp execution for chess piece at RANDOM positions.

The piece spawns laying down at a random (x, y, yaw) on the table.
All grasp geometry is computed from the ACTUAL settled pose — fully invariant.

Workflow:
  ENTER → PRE-GRASP → ENTER → GRASP → ENTER → CLOSE → ENTER → LIFT

Markers:
  GREEN   = Target fingertip positions
  MAGENTA = Actual fingertip tips (on finger bodies)
  YELLOW  = Grasp center
  CYAN    = Pre-grasp hover point
"""
import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import threading
import sys

# ── Scene config ──────────────────────────────────────────────
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
OUTPUT_XML = "franka_emika_panda/grasp_ik_scene.xml"

HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float)

# Grasp geometry
GRASP_OFFSET = 0.103
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.12

# IK parameters
IK_TOL = 1e-3
IK_MAX_ITER = 200
IK_STEP = 0.3
IK_DAMPING = 1e-4

SETTLE_DURATION = 1.0

# ── Piece spawn randomization ────────────────────────────────
# Reachable area on the table in front of the robot
SPAWN_X_RANGE = (0.35, 0.55)   # forward/back
SPAWN_Y_RANGE = (-0.15, 0.15)  # left/right
SPAWN_Z = 0.032                 # height (laying on table)
SPAWN_YAW_RANGE = (-1, 1)  # radians of yaw variation while laying down


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
    """Axis-angle to quaternion [w,x,y,z]."""
    axis = axis / np.linalg.norm(axis)
    ha = angle / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


# ══════════════════════════════════════════════════════════════
#  Scene building
# ══════════════════════════════════════════════════════════════

def random_piece_pose():
    """Generate random (pos, quat) for piece laying on its side."""
    x = np.random.uniform(*SPAWN_X_RANGE)
    y = np.random.uniform(*SPAWN_Y_RANGE)
    yaw = np.random.uniform(*SPAWN_YAW_RANGE)

    # Base orientation: 90° around Y (laying on side)
    q_lay = np.array([0.7071068, 0, 0.7071068, 0])

    # Add yaw rotation around world Z
    q_yaw = axis_angle_to_quat(np.array([0, 0, 1]), yaw)
    q_final = quat_multiply(q_yaw, q_lay)

    pos = [x, y, SPAWN_Z]
    return pos, q_final, yaw


def find_finger_body(root, finger_name):
    for body in root.iter("body"):
        if body.get("name") == finger_name:
            return body
    return None


def build_scene(piece_pos, piece_quat):
    """Build XML scene with chess piece at given pose."""
    modelTree = ET.parse(ROOT_MODEL_XML)
    root = modelTree.getroot()

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {
        "name": "chess_king_mesh",
        "file": "chess_king.stl",
        "scale": "0.001 0.001 0.001"
    })

    worldbody = root.find("worldbody")

    pq = piece_quat
    body = ET.SubElement(worldbody, "body", {
        "name": "ChessPiece",
        "pos": f"{piece_pos[0]} {piece_pos[1]} {piece_pos[2]}",
        "quat": f"{pq[0]} {pq[1]} {pq[2]} {pq[3]}"
    })
    ET.SubElement(body, "freejoint", {"name": "chess_piece_joint"})

    # Visual mesh (no collision)
    ET.SubElement(body, "geom", {
        "name": "chess_piece_visual",
        "type": "mesh",
        "mesh": "chess_king_mesh",
        "rgba": "0.85 0.75 0.55 1",
        "contype": "0", "conaffinity": "0",
        "mass": "0"
    })

    # Collision box for base (where fingers grip)
    ET.SubElement(body, "geom", {
        "name": "chess_piece_base_col",
        "type": "box",
        "size": "0.016 0.016 0.016",
        "pos": "0.016 0.016 0.016",
        "rgba": "0.85 0.75 0.55 0.3",
        "density": "500",
        "friction": "1.5 1.5 0.5",
        "contype": "1", "conaffinity": "1",
        "condim": "4"
    })

    # Collision cylinder for stem
    ET.SubElement(body, "geom", {
        "name": "chess_piece_stem_col",
        "type": "cylinder",
        "size": "0.010 0.030",
        "pos": "0.016 0.016 0.060",
        "rgba": "0.85 0.75 0.55 0.2",
        "density": "500",
        "friction": "1.5 0.5 0.1",
        "contype": "1", "conaffinity": "1",
        "condim": "4"
    })

    # Magenta fingertip trackers on finger bodies
    left_finger_body = find_finger_body(root, "left_finger")
    right_finger_body = find_finger_body(root, "right_finger")
    if left_finger_body is not None:
        ET.SubElement(left_finger_body, "geom", {
            "name": "actual_left_tip",
            "type": "sphere", "size": "0.012",
            "pos": "0 0 0.045",
            "rgba": "1 0 1 0.9",
            "contype": "0", "conaffinity": "0"
        })
    if right_finger_body is not None:
        ET.SubElement(right_finger_body, "geom", {
            "name": "actual_right_tip",
            "type": "sphere", "size": "0.012",
            "pos": "0 0 0.045",
            "rgba": "1 0 1 0.9",
            "contype": "0", "conaffinity": "0"
        })

    # NOTE: We add marker geoms AFTER settling, so we write a base scene first
    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


def add_markers_to_scene(grasp_center, target_left, target_right, pre_grasp_tip):
    """Add visualization markers to the already-written XML."""
    modelTree = ET.parse(OUTPUT_XML)
    root = modelTree.getroot()
    worldbody = root.find("worldbody")

    ET.SubElement(worldbody, "geom", {
        "name": "target_left",
        "type": "sphere", "size": "0.012",
        "pos": f"{target_left[0]} {target_left[1]} {target_left[2]}",
        "rgba": "0 1 0 0.9",
        "contype": "0", "conaffinity": "0"
    })
    ET.SubElement(worldbody, "geom", {
        "name": "target_right",
        "type": "sphere", "size": "0.012",
        "pos": f"{target_right[0]} {target_right[1]} {target_right[2]}",
        "rgba": "0 1 0 0.9",
        "contype": "0", "conaffinity": "0"
    })
    ET.SubElement(worldbody, "geom", {
        "name": "grasp_center_marker",
        "type": "sphere", "size": "0.008",
        "pos": f"{grasp_center[0]} {grasp_center[1]} {grasp_center[2]}",
        "rgba": "1 1 0 0.9",
        "contype": "0", "conaffinity": "0"
    })
    ET.SubElement(worldbody, "geom", {
        "name": "pre_grasp_marker",
        "type": "sphere", "size": "0.008",
        "pos": f"{pre_grasp_tip[0]} {pre_grasp_tip[1]} {pre_grasp_tip[2]}",
        "rgba": "0 1 1 0.9",
        "contype": "0", "conaffinity": "0"
    })

    modelTree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)


# ══════════════════════════════════════════════════════════════
#  Grasp geometry — computed from ACTUAL settled piece pose
# ══════════════════════════════════════════════════════════════

def calculate_grasp_from_pose(piece_pos, piece_quat):
    """
    Compute grasp geometry from the piece's actual world pose.

    The base cube center in LOCAL frame is at (16, 16, 16)mm.
    We rotate that offset by the piece's world quaternion to get
    the grasp center in world frame.

    Finger axis is perpendicular to the piece's laying direction,
    determined from the piece's rotated local Y axis.
    """
    R_piece = quat_to_rotmat(piece_quat)

    # Base center in local frame (meters)
    local_base_center = np.array([0.016, 0.016, 0.016])

    # Grasp center in world frame
    grasp_center = piece_pos + R_piece @ local_base_center

    # Finger axis: local Y rotated to world
    finger_axis = R_piece @ np.array([0.0, 1.0, 0.0])
    finger_axis = finger_axis / np.linalg.norm(finger_axis)

    # Face positions: ±16mm along finger axis from grasp center
    face1_pos = grasp_center + 0.016 * finger_axis
    face2_pos = grasp_center - 0.016 * finger_axis

    # Target fingertip positions: ±GRIPPER_OPEN along finger axis
    target_left = grasp_center + GRIPPER_OPEN * finger_axis
    target_right = grasp_center - GRIPPER_OPEN * finger_axis

    # Gripper orientation: Z points DOWN, Y along finger axis
    gripper_z = np.array([0, 0, -1])  # points down
    gripper_y = -finger_axis           # Franka convention: -Y is spread direction
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x = gripper_x / np.linalg.norm(gripper_x)
    # Re-orthogonalize
    gripper_y = np.cross(gripper_z, gripper_x)

    R_gripper = np.column_stack([gripper_x, gripper_y, gripper_z])
    gripper_quat = rotmat_to_quat(R_gripper)

    # Waypoints (fingertip targets)
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


# ══════════════════════════════════════════════════════════════
#  IK solver
# ══════════════════════════════════════════════════════════════

def compute_ik(model, data, target_pos, target_quat, q_init):
    hand_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    nv = 7

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
            print(f"  IK converged in {i} iters, pos_err={err_norm:.6f}")
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
        d.qpos[:nv] += IK_STEP * dq

        for j in range(nv):
            lo = model.jnt_range[j, 0]
            hi = model.jnt_range[j, 1]
            if lo < hi:
                d.qpos[j] = np.clip(d.qpos[j], lo, hi)

        mj.mj_forward(model, d)

    print(f"  IK did NOT converge after {IK_MAX_ITER} iters, pos_err={err_norm:.6f}")
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
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Optional: pass a seed for reproducibility
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")

    print("=" * 60)
    print("  IK GRASP — RANDOM PIECE POSITION")
    print("=" * 60)

    # ── Random piece pose ─────────────────────────────────────
    piece_pos, piece_quat, yaw = random_piece_pose()
    print(f"\nSpawn position: [{piece_pos[0]:.4f}, {piece_pos[1]:.4f}, {piece_pos[2]:.4f}]")
    print(f"Spawn quat:     [{piece_quat[0]:.4f}, {piece_quat[1]:.4f}, {piece_quat[2]:.4f}, {piece_quat[3]:.4f}]")
    print(f"Yaw offset:     {np.degrees(yaw):.1f}°")

    # ── Build base scene (no markers yet) ─────────────────────
    build_scene(piece_pos, piece_quat)

    # ── Load and settle ───────────────────────────────────────
    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    mj.mj_forward(model, data)

    print("\nSettling piece...")
    for _ in range(3000):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        data.ctrl[:7] = KP * (HOME_Q - q) + KD * (0 - qd) + data.qfrc_bias[:7]
        data.ctrl[7] = GRIPPER_OPEN
        mj.mj_step(model, data)

    # ── Read ACTUAL settled pose ──────────────────────────────
    # freejoint qpos: [x,y,z, qw,qx,qy,qz] starting after 7 arm + 2 gripper = index 9
    settled_pos = data.qpos[9:12].copy()
    settled_quat = data.qpos[12:16].copy()

    print(f"\nSettled position: [{settled_pos[0]:.4f}, {settled_pos[1]:.4f}, {settled_pos[2]:.4f}]")
    print(f"Settled quat:     [{settled_quat[0]:.4f}, {settled_quat[1]:.4f}, {settled_quat[2]:.4f}, {settled_quat[3]:.4f}]")

    # ── Compute grasp geometry from settled pose ──────────────
    geom = calculate_grasp_from_pose(settled_pos, settled_quat)

    print(f"\nGrasp center:   {geom['grasp_center']}")
    print(f"Finger axis:    {geom['finger_axis']}")
    print(f"Target LEFT:    {geom['target_left']}")
    print(f"Target RIGHT:   {geom['target_right']}")
    print(f"Pre-grasp tip:  {geom['pre_grasp_tip']}")
    print(f"Grasp tip:      {geom['grasp_tip']}")
    print(f"Lift tip:       {geom['lift_tip']}")

    # ── Add markers and reload ────────────────────────────────
    add_markers_to_scene(
        geom["grasp_center"],
        geom["target_left"],
        geom["target_right"],
        geom["pre_grasp_tip"]
    )

    # Reload model with markers
    model = mj.MjModel.from_xml_path(OUTPUT_XML)
    data = mj.MjData(model)
    data.qpos[:7] = HOME_Q
    data.qpos[7] = GRIPPER_OPEN
    data.qpos[8] = GRIPPER_OPEN
    # Restore settled piece pose
    data.qpos[9:12] = settled_pos
    data.qpos[12:16] = settled_quat
    mj.mj_forward(model, data)

    # ── Solve IK ──────────────────────────────────────────────
    target_quat = geom["gripper_quat"]

    print("\n--- Solving IK for PRE-GRASP ---")
    q_pre_grasp, ok1 = compute_ik(model, data, geom["pre_grasp_tip"], target_quat, HOME_Q)

    print("\n--- Solving IK for GRASP ---")
    q_grasp, ok2 = compute_ik(model, data, geom["grasp_tip"], target_quat, q_pre_grasp)

    print("\n--- Solving IK for LIFT ---")
    q_lift, ok3 = compute_ik(model, data, geom["lift_tip"], target_quat, q_grasp)

    if not (ok1 and ok2 and ok3):
        print("\nWARNING: Not all IK solutions converged!")

    print(f"\nIK solutions (rad):")
    print(f"  Pre-grasp: {np.round(q_pre_grasp, 4)}")
    print(f"  Grasp:     {np.round(q_grasp, 4)}")
    print(f"  Lift:      {np.round(q_lift, 4)}")

    # ── Build trajectories ────────────────────────────────────
    traj_steps = 60
    traj_to_pre = interpolate_joints(HOME_Q, q_pre_grasp, traj_steps)
    traj_to_grasp = interpolate_joints(q_pre_grasp, q_grasp, traj_steps)
    traj_to_lift = interpolate_joints(q_grasp, q_lift, traj_steps)

    # ── Launch viewer ─────────────────────────────────────────
    v = viewer.launch_passive(model, data)
    v.cam.distance = 0.9
    v.cam.azimuth = 135
    v.cam.elevation = -30
    v.cam.lookat[:] = [settled_pos[0], settled_pos[1], 0.15]

    # ── Terminal input thread ─────────────────────────────────
    advance_event = threading.Event()

    def input_thread_fn():
        while True:
            try:
                input()
                advance_event.set()
            except EOFError:
                break

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)
    input_thread.start()

    print("\n" + "=" * 60)
    print("  VIEWER LAUNCHED — press ENTER in terminal to advance")
    print("=" * 60)
    print("\n>>> Press ENTER to move to PRE-GRASP")

    phase = 0

    try:
        while v.is_running():
            if phase == 0:
                hold_position(model, data, v, HOME_Q, GRIPPER_OPEN, 0.05)
                if advance_event.is_set():
                    advance_event.clear()
                    print("\n>>> Moving to PRE-GRASP...")
                    execute_trajectory(model, data, v, traj_to_pre, GRIPPER_OPEN)
                    hold_position(model, data, v, q_pre_grasp, GRIPPER_OPEN, SETTLE_DURATION)
                    phase = 1
                    print(">>> AT PRE-GRASP. Press ENTER to lower to GRASP")

            elif phase == 1:
                hold_position(model, data, v, q_pre_grasp, GRIPPER_OPEN, 0.05)
                if advance_event.is_set():
                    advance_event.clear()
                    print("\n>>> Lowering to GRASP...")
                    execute_trajectory(model, data, v, traj_to_grasp, GRIPPER_OPEN)
                    hold_position(model, data, v, q_grasp, GRIPPER_OPEN, SETTLE_DURATION)
                    phase = 2
                    print(">>> AT GRASP. Press ENTER to CLOSE gripper")

            elif phase == 2:
                hold_position(model, data, v, q_grasp, GRIPPER_OPEN, 0.05)
                if advance_event.is_set():
                    advance_event.clear()
                    print("\n>>> Closing gripper...")
                    hold_position(model, data, v, q_grasp, GRIPPER_CLOSED, 2.0)
                    phase = 3
                    print(">>> GRASPED. Press ENTER to LIFT")

            elif phase == 3:
                hold_position(model, data, v, q_grasp, GRIPPER_CLOSED, 0.05)
                if advance_event.is_set():
                    advance_event.clear()
                    print("\n>>> Lifting...")
                    execute_trajectory(model, data, v, traj_to_lift, GRIPPER_CLOSED)
                    hold_position(model, data, v, q_lift, GRIPPER_CLOSED, SETTLE_DURATION)
                    phase = 4
                    print(">>> DONE. Holding. Close viewer to exit.")
                    print(">>> Or press ENTER to restart with a new random position.")

            elif phase == 4:
                hold_position(model, data, v, q_lift, GRIPPER_CLOSED, 0.05)
                if advance_event.is_set():
                    advance_event.clear()
                    print("\n>>> RESTARTING with new random position...")
                    v.close()
                    break

    except KeyboardInterrupt:
        v.close()
        print("\nDone.")
        sys.exit(0)

    # ── If we broke out of loop, restart ──────────────────────
    v.close()
    print("\nRe-run the script for a new random position.")
    print("Tip: pass a seed number as argument for reproducibility:")
    print("  mjpython grasp_ik.py 42")