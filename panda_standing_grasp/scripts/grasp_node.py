#!/usr/bin/env python
"""
ROS1 node: top-down grasp of a standing chess piece with Franka Panda.

Subscribes:
  /piece_pose   (geometry_msgs/PoseStamped)  — center of piece, Z up along piece
  /place_target (geometry_msgs/PointStamped)  — desired placement position
  /go           (std_msgs/Empty)             — publish to advance to next step

Publishes:
  /grasp_waypoints (visualization_msgs/MarkerArray) — labeled waypoints for RViz
  /current_step    (std_msgs/String)                — name of the step being executed

Flow:
  Receives piece_pose + place_target → plans all waypoints.
  Then waits for /go before each step:
    1. HOME        (open gripper, move to home)
    2. PRE-GRASP   (move above piece)
    3. GRASP       (descend to stem)
    4. CLOSE       (close gripper)
    5. LIFT        (raise piece)
    6. PRE-PLACE   (move above target)
    7. PLACE       (lower piece)
    8. OPEN        (release)
    9. RETREAT     (back away)
"""
import threading
import rospy
import numpy as np
import actionlib

from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Empty, String, ColorRGBA, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from visualization_msgs.msg import Marker, MarkerArray
from franka_gripper.msg import (
    GraspAction, GraspGoal, GraspEpsilon,
    MoveAction, MoveGoal,
)

from panda_standing_grasp.planner import GraspPlanner


JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]


class GraspNode:
    def __init__(self):
        rospy.init_node("panda_standing_grasp")

        # ── Load params ──
        self.cfg = {
            "grasp_z_offset_from_center": rospy.get_param("~grasp_z_offset_from_center", 0.015),
            "stem_radius": rospy.get_param("~stem_radius", 0.010),
            "gripper_open_width": rospy.get_param("~gripper_open_width", 0.04),
            "gripper_close_width": rospy.get_param("~gripper_close_width", 0.005),
            "grasp_force": rospy.get_param("~grasp_force", 5.0),
            "grasp_epsilon_inner": rospy.get_param("~grasp_epsilon_inner", 0.005),
            "grasp_epsilon_outer": rospy.get_param("~grasp_epsilon_outer", 0.005),
            "pre_grasp_height": rospy.get_param("~pre_grasp_height", 0.10),
            "lift_height": rospy.get_param("~lift_height", 0.15),
            "pre_place_offset": rospy.get_param("~pre_place_offset", 0.10),
            "grasp_offset": rospy.get_param("~grasp_offset", 0.103),
            "ik_tol": rospy.get_param("~ik_tol", 0.001),
            "ik_max_iter": rospy.get_param("~ik_max_iter", 200),
            "trajectory_velocity_scale": rospy.get_param("~trajectory_velocity_scale", 0.3),
            "hold_duration": rospy.get_param("~hold_duration", 2.0),
            "table_height": rospy.get_param("~table_height", 0.3),
            "table_half_size": rospy.get_param("~table_half_size", 0.15),
            "collision_margin": rospy.get_param("~collision_margin", 0.05),
            "home_q": rospy.get_param("~home_q", [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]),
        }

        model_xml = rospy.get_param("~model_xml")
        self.planner = GraspPlanner(model_xml, self.cfg)

        self.vel_scale = self.cfg["trajectory_velocity_scale"]
        self.hold_dur = self.cfg["hold_duration"]

        # ── Received data ──
        self.piece_pose = None
        self.place_target = None

        # ── /go gate ──
        self._go_event = threading.Event()

        # ── Subscribers ──
        piece_topic = rospy.get_param("~piece_pose_topic", "/piece_pose")
        place_topic = rospy.get_param("~place_target_topic", "/place_target")
        rospy.Subscriber(piece_topic, PoseStamped, self._piece_cb)
        rospy.Subscriber(place_topic, PointStamped, self._place_cb)
        rospy.Subscriber("/go", Empty, self._go_cb)

        # ── Publishers ──
        self.marker_pub = rospy.Publisher(
            "/grasp_waypoints", MarkerArray, queue_size=1, latch=True)
        self.step_pub = rospy.Publisher(
            "/current_step", String, queue_size=1, latch=True)

        # ── Action clients ──
        arm_action = rospy.get_param(
            "~arm_trajectory_action",
            "/position_joint_trajectory_controller/follow_joint_trajectory")
        gripper_grasp_action = rospy.get_param(
            "~gripper_grasp_action", "/franka_gripper/grasp")
        gripper_move_action = rospy.get_param(
            "~gripper_move_action", "/franka_gripper/move")

        rospy.loginfo("Connecting to arm action server: %s", arm_action)
        self.arm_client = actionlib.SimpleActionClient(
            arm_action, FollowJointTrajectoryAction)

        rospy.loginfo("Connecting to gripper grasp action: %s", gripper_grasp_action)
        self.gripper_grasp_client = actionlib.SimpleActionClient(
            gripper_grasp_action, GraspAction)

        rospy.loginfo("Connecting to gripper move action: %s", gripper_move_action)
        self.gripper_move_client = actionlib.SimpleActionClient(
            gripper_move_action, MoveAction)

        self.arm_client.wait_for_server(rospy.Duration(10))
        self.gripper_grasp_client.wait_for_server(rospy.Duration(10))
        self.gripper_move_client.wait_for_server(rospy.Duration(10))
        rospy.loginfo("All action servers connected.")

    # ── Callbacks ──────────────────────────────────────────────

    def _piece_cb(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        self.piece_pose = {
            "pos": np.array([p.x, p.y, p.z]),
            "quat": np.array([o.w, o.x, o.y, o.z]),
        }
        rospy.loginfo("Received piece pose: [%.3f, %.3f, %.3f]", p.x, p.y, p.z)

    def _place_cb(self, msg):
        p = msg.point
        self.place_target = np.array([p.x, p.y, p.z])
        rospy.loginfo("Received place target: [%.3f, %.3f, %.3f]", p.x, p.y, p.z)

    def _go_cb(self, msg):
        self._go_event.set()

    # ── Step gate ──────────────────────────────────────────────

    def _wait_for_go(self, step_name):
        """Block until a message arrives on /go. Publishes step name."""
        self.step_pub.publish(String(data=step_name))
        rospy.loginfo(">> Waiting for /go to execute: %s", step_name)
        while not rospy.is_shutdown():
            if self._go_event.wait(timeout=0.2):
                self._go_event.clear()
                return True
        return False

    # ── Visualization ──────────────────────────────────────────

    def _publish_waypoint_markers(self, plan):
        ma = MarkerArray()
        g = plan["grasp_data"]
        p = plan["place_data"]
        stamp = rospy.Time.now()
        frame = "world"

        def _sphere(mid, pos, r, gv, b, a=0.9, scale=0.016, ns="waypoints"):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=frame)
            m.ns, m.id = ns, mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = ColorRGBA(r=r, g=gv, b=b, a=a)
            return m

        def _line(mid, points, r, gv, b, a=0.6, width=0.004, ns="paths"):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=frame)
            m.ns, m.id = ns, mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = width
            m.color = ColorRGBA(r=r, g=gv, b=b, a=a)
            for pt in points:
                m.points.append(Point(x=pt[0], y=pt[1], z=pt[2]))
            return m

        def _text(mid, pos, text, ns="labels"):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=frame)
            m.ns, m.id = ns, mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position = Point(x=pos[0], y=pos[1], z=pos[2] + 0.025)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.02
            m.color = ColorRGBA(r=1, g=1, b=1, a=0.9)
            m.text = text
            return m

        mid = 0
        for label, pos, rgb, sz in [
            ("PRE-GRASP",  g["pre_grasp_tip"],  (0, 1, 1),     0.016),
            ("GRASP",      g["grasp_center"],    (1, 1, 0),     0.020),
            ("L-finger",   g["target_left"],     (0, 1, 0),     0.012),
            ("R-finger",   g["target_right"],    (0, 1, 0),     0.012),
            ("LIFT",       g["lift_tip"],         (1, 0, 1),     0.016),
            ("PRE-PLACE",  p["pre_place_tip"],   (1, 0.5, 0),   0.016),
            ("PLACE",      p["place_tip"],        (1, 0, 0),     0.016),
            ("RETREAT",    p["retreat_tip"],       (1, 1, 1),     0.016),
        ]:
            ma.markers.append(_sphere(mid, pos, *rgb, scale=sz)); mid += 1
            ma.markers.append(_text(mid, pos, label)); mid += 1

        n = 10
        for pts, rgb in [
            ([g["pre_grasp_tip"], g["grasp_tip"]],     (0.3, 0.3, 1)),
            ([g["grasp_tip"], g["lift_tip"]],           (1, 0, 1)),
            ([g["lift_tip"], p["pre_place_tip"]],       (1, 0.6, 0.8)),
            ([p["pre_place_tip"], p["place_tip"]],      (1, 0.5, 0)),
        ]:
            interp = [pts[0] + (k / n) * (pts[1] - pts[0]) for k in range(n + 1)]
            ma.markers.append(_line(mid, interp, *rgb)); mid += 1

        fa = g["finger_axis"]
        gc = g["grasp_center"]
        w = self.cfg["gripper_open_width"]
        ma.markers.append(_line(mid, [gc - w * fa, gc + w * fa], 0, 0.8, 0, width=0.003)); mid += 1

        self.marker_pub.publish(ma)
        rospy.loginfo("Published %d waypoint markers to /grasp_waypoints", len(ma.markers))

    # ── Robot commands ─────────────────────────────────────────

    def _send_trajectory(self, waypoints, label=""):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES
        t = 0.0
        for i, wp in enumerate(waypoints):
            pt = JointTrajectoryPoint()
            pt.positions = wp.tolist()
            pt.velocities = [0.0] * 7
            if i > 0:
                dq = np.max(np.abs(wp - waypoints[i - 1]))
                t += max(dq / self.vel_scale, 0.01)
            pt.time_from_start = rospy.Duration.from_sec(t)
            traj_msg.points.append(pt)

        goal = FollowJointTrajectoryGoal(trajectory=traj_msg)
        rospy.loginfo("  Executing: %s (%d pts, %.1fs)", label,
                      len(waypoints), t)
        self.arm_client.send_goal(goal)
        self.arm_client.wait_for_result()

    def _gripper_close(self):
        goal = GraspGoal()
        goal.width = self.cfg["gripper_close_width"]
        goal.speed = 0.04
        goal.force = self.cfg["grasp_force"]
        goal.epsilon = GraspEpsilon(
            inner=self.cfg["grasp_epsilon_inner"],
            outer=self.cfg["grasp_epsilon_outer"])
        rospy.loginfo("  Closing gripper (force=%.1f N)", goal.force)
        self.gripper_grasp_client.send_goal(goal)
        self.gripper_grasp_client.wait_for_result(rospy.Duration(5))

    def _gripper_open(self):
        goal = MoveGoal()
        goal.width = self.cfg["gripper_open_width"]
        goal.speed = 0.04
        rospy.loginfo("  Opening gripper")
        self.gripper_move_client.send_goal(goal)
        self.gripper_move_client.wait_for_result(rospy.Duration(5))

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        rospy.loginfo("Waiting for /piece_pose ...")
        while not rospy.is_shutdown() and self.piece_pose is None:
            rospy.sleep(0.1)

        rospy.loginfo("Waiting for /place_target ...")
        while not rospy.is_shutdown() and self.place_target is None:
            rospy.sleep(0.1)

        if rospy.is_shutdown():
            return

        rospy.loginfo("Planning full grasp sequence...")
        plan = self.planner.plan_full_grasp(
            self.piece_pose["pos"],
            self.piece_pose["quat"],
            self.place_target,
        )

        if not plan["ik_ok"]:
            labels = ["pre_grasp", "grasp", "lift", "pre_place", "place", "retreat"]
            for lbl, ok in zip(labels, plan["ik_results"]):
                if not ok:
                    rospy.logwarn("  IK FAILED for: %s", lbl)

        self._publish_waypoint_markers(plan)
        trajs = plan["trajs"]

        steps = [
            ("1/9  HOME",       lambda: (self._gripper_open(),
                                         self._send_trajectory(trajs["to_home"], "HOME"))),
            ("2/9  PRE-GRASP",  lambda: self._send_trajectory(trajs["to_pre_grasp"], "PRE-GRASP")),
            ("3/9  GRASP",      lambda: self._send_trajectory(trajs["to_grasp"], "GRASP")),
            ("4/9  CLOSE",      lambda: self._gripper_close()),
            ("5/9  LIFT",       lambda: self._send_trajectory(trajs["to_lift"], "LIFT")),
            ("6/9  PRE-PLACE",  lambda: self._send_trajectory(trajs["to_pre_place"], "PRE-PLACE")),
            ("7/9  PLACE",      lambda: self._send_trajectory(trajs["to_place"], "PLACE")),
            ("8/9  OPEN",       lambda: self._gripper_open()),
            ("9/9  RETREAT",    lambda: self._send_trajectory(trajs["to_retreat"], "RETREAT")),
        ]

        rospy.loginfo("Plan ready. Publish to /go (std_msgs/Empty) to advance each step.")

        for step_name, action in steps:
            if not self._wait_for_go(step_name):
                return
            action()
            rospy.loginfo("  %s done.", step_name)

        self.step_pub.publish(String(data="DONE"))
        rospy.loginfo("All steps complete.")


if __name__ == "__main__":
    try:
        node = GraspNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
