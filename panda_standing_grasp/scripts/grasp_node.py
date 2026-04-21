#!/usr/bin/env python
"""
ROS1 node: top-down grasp of a standing chess piece with Franka Panda.

Uses frankapy for all robot control (no franka_ros / action clients needed).

Subscribes:
  /piece_pose   (geometry_msgs/PoseStamped)  — center of piece, Z up along piece
  /place_target (geometry_msgs/PointStamped)  — desired placement position
  /go           (std_msgs/Empty)             — publish to advance to next step

Publishes:
  /grasp_waypoints (visualization_msgs/MarkerArray) — labeled waypoints for RViz
  /current_step    (std_msgs/String)                — name of the step being executed

Flow:
  Receives piece_pose + place_target -> plans all waypoints.
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

from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Empty, String, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from frankapy import FrankaArm

from panda_standing_grasp.planner import GraspPlanner


class GraspNode:
    def __init__(self):
        rospy.init_node("panda_standing_grasp")

        # -- Load params --
        self.cfg = {
            "grasp_z_offset_from_center": rospy.get_param("~grasp_z_offset_from_center", 0.015),
            "stem_radius": rospy.get_param("~stem_radius", 0.010),
            "gripper_open_width": rospy.get_param("~gripper_open_width", 0.04),
            "gripper_close_width": rospy.get_param("~gripper_close_width", 0.005),
            "pre_grasp_height": rospy.get_param("~pre_grasp_height", 0.10),
            "lift_height": rospy.get_param("~lift_height", 0.15),
            "pre_place_offset": rospy.get_param("~pre_place_offset", 0.10),
            "grasp_offset": rospy.get_param("~grasp_offset", 0.103),
            "ik_tol": rospy.get_param("~ik_tol", 0.001),
            "ik_max_iter": rospy.get_param("~ik_max_iter", 200),
            "goto_duration": rospy.get_param("~goto_duration", 5.0),
            "table_height": rospy.get_param("~table_height", 0.3),
            "table_half_size": rospy.get_param("~table_half_size", 0.15),
            "collision_margin": rospy.get_param("~collision_margin", 0.05),
            "home_q": rospy.get_param("~home_q", [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]),
        }

        model_xml = rospy.get_param("~model_xml")
        self.planner = GraspPlanner(model_xml, self.cfg)

        self.goto_dur = self.cfg["goto_duration"]

        # -- Connect to FrankaArm --
        rospy.loginfo("Connecting to FrankaArm...")
        self.fa = FrankaArm()
        rospy.loginfo("FrankaArm connected.")

        # -- Received data --
        self.piece_pose = None
        self.place_target = None

        # -- /go gate --
        self._go_event = threading.Event()

        # -- Subscribers --
        piece_topic = rospy.get_param("~piece_pose_topic", "/piece_pose")
        place_topic = rospy.get_param("~place_target_topic", "/place_target")
        rospy.Subscriber(piece_topic, PoseStamped, self._piece_cb)
        rospy.Subscriber(place_topic, PointStamped, self._place_cb)
        rospy.Subscriber("/go", Empty, self._go_cb)

        # -- Publishers --
        self.marker_pub = rospy.Publisher(
            "/grasp_waypoints", MarkerArray, queue_size=1, latch=True)
        self.step_pub = rospy.Publisher(
            "/current_step", String, queue_size=1, latch=True)

        rospy.loginfo("Grasp node ready (frankapy mode).")

    # -- Callbacks --

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

    # -- Step gate --

    def _wait_for_go(self, step_name):
        """Block until a message arrives on /go. Publishes step name."""
        self.step_pub.publish(String(data=step_name))
        rospy.loginfo(">> Waiting for /go to execute: %s", step_name)
        while not rospy.is_shutdown():
            if self._go_event.wait(timeout=0.2):
                self._go_event.clear()
                return True
        return False

    # -- Visualization --

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

        fa_axis = g["finger_axis"]
        gc = g["grasp_center"]
        w = self.cfg["gripper_open_width"]
        ma.markers.append(_line(mid, [gc - w * fa_axis, gc + w * fa_axis], 0, 0.8, 0, width=0.003)); mid += 1

        self.marker_pub.publish(ma)
        rospy.loginfo("Published %d waypoint markers to /grasp_waypoints", len(ma.markers))

    # -- Robot commands (frankapy) --

    def _goto_joints(self, q, label=""):
        """Send robot to a joint configuration via frankapy."""
        rospy.loginfo("  Executing: %s -> joints", label)
        self.fa.goto_joints(q.tolist(), duration=self.goto_dur)

    def _gripper_close(self):
        rospy.loginfo("  Closing gripper")
        self.fa.close_gripper()

    def _gripper_open(self):
        rospy.loginfo("  Opening gripper")
        self.fa.open_gripper()

    # -- Main loop --

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
        jc = plan["joint_configs"]

        steps = [
            ("1/9  HOME",       lambda: (self._gripper_open(),
                                         self._goto_joints(jc["home"], "HOME"))),
            ("2/9  PRE-GRASP",  lambda: self._goto_joints(jc["pre_grasp"], "PRE-GRASP")),
            ("3/9  GRASP",      lambda: self._goto_joints(jc["grasp"], "GRASP")),
            ("4/9  CLOSE",      lambda: self._gripper_close()),
            ("5/9  LIFT",       lambda: self._goto_joints(jc["lift"], "LIFT")),
            ("6/9  PRE-PLACE",  lambda: self._goto_joints(jc["pre_place"], "PRE-PLACE")),
            ("7/9  PLACE",      lambda: self._goto_joints(jc["place"], "PLACE")),
            ("8/9  OPEN",       lambda: self._gripper_open()),
            ("9/9  RETREAT",    lambda: self._goto_joints(jc["retreat"], "RETREAT")),
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
