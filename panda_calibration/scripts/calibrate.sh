#!/bin/bash
# Usage: bash calibrate.sh [num_samples]
# Run AFTER:
#   1. roscore is running
#   2. frankapy server is running (start_control_pc.sh)
#   3. RealSense camera is plugged in
# No franka_control / franka_ros needed.

NUM_SAMPLES=${1:-12}

echo "============================================"
echo "  HAND-EYE CALIBRATION (frankapy)"
echo "  Samples: $NUM_SAMPLES"
echo "============================================"
echo ""
echo "Prerequisites:"
echo "  1. roscore running"
echo "  2. frankapy server running (start_control_pc.sh)"
echo "  3. Robot lights are BLUE"
echo "  4. catkin_ws built and sourced"
echo "============================================"

cleanup() {
    echo "Shutting down..."
    kill 0 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

source ~/catkin_ws/devel/setup.bash

# Launch camera + calibration node + RViz (no robot driver needed)
roslaunch panda_calibration calibrate.launch &
sleep 5

echo ""
echo "============================================"
echo "  READY — move robot, press ENTER to capture"
echo "============================================"
echo ""

for i in $(seq 1 $NUM_SAMPLES); do
    read -p "[$i/$NUM_SAMPLES] Move robot to new pose → ENTER to capture..."
    rostopic pub --once /calibrate/capture std_msgs/Empty "{}"
    sleep 1
done

echo ""
echo "Solving..."
rostopic pub --once /calibrate/solve std_msgs/Empty "{}"
sleep 2

echo ""
rostopic echo -n 1 /calibration/status
echo ""
echo "Result saved to ~/calibration_result.yaml"
echo "Press Ctrl+C to exit."

wait
