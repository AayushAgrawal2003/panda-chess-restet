#!/bin/bash
# ── One-shot calibration launcher ──
# Usage: bash calibrate.sh [control-pc-name]
#
# Launches everything, then you just move the robot and press ENTER to capture.

set -e

CONTROL_PC=${1:?"Usage: bash calibrate.sh <control-pc-name>"}
ROBOT_IP=${2:-172.16.0.2}
NUM_SAMPLES=${3:-12}

echo "============================================"
echo "  HAND-EYE CALIBRATION"
echo "  Control PC : $CONTROL_PC"
echo "  Robot IP   : $ROBOT_IP"
echo "  Samples    : $NUM_SAMPLES"
echo "============================================"

cleanup() {
    echo "Shutting down..."
    kill 0 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# ── 1. roscore ──
echo "[1/4] Starting roscore..."
roscore &
sleep 3

# ── 2. Robot driver ──
echo "[2/4] Starting franka_control..."
roslaunch franka_control franka_control.launch robot_ip:=$ROBOT_IP &
sleep 5

# ── 3. Camera ──
echo "[3/4] Starting RealSense..."
roslaunch realsense2_camera rs_camera.launch align_depth:=true &
sleep 3

# ── 4. Calibration node + RViz ──
echo "[4/4] Starting calibration node + RViz..."
source ~/catkin_ws/devel/setup.bash
roslaunch panda_calibration calibrate.launch launch_camera:=false &
sleep 3

echo ""
echo "============================================"
echo "  READY"
echo ""
echo "  1. Move the robot (guide mode: e-stop → squeeze wrist → move)"
echo "  2. Release e-stop (twist) so lights are BLUE"
echo "  3. Press ENTER here to capture a sample"
echo "  4. Repeat for $NUM_SAMPLES poses"
echo "  5. Calibration runs automatically after $NUM_SAMPLES samples"
echo "============================================"
echo ""

for i in $(seq 1 $NUM_SAMPLES); do
    read -p "[$i/$NUM_SAMPLES] Move robot to new pose, then press ENTER..."
    rostopic pub --once /calibrate/capture std_msgs/Empty "{}"
    echo "  Captured sample $i."
    echo ""
done

echo "Solving calibration..."
rostopic pub --once /calibrate/solve std_msgs/Empty "{}"
sleep 2

echo ""
echo "============================================"
rostopic echo -n 1 /calibration/status
echo "============================================"
echo ""
echo "Result saved to ~/calibration_result.yaml"
echo "TF is live in RViz. Press Ctrl+C to exit."

wait
