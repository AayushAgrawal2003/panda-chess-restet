cmake_minimum_required(VERSION 3.0.2)
project(panda_calibration)

find_package(catkin REQUIRED COMPONENTS
  rospy std_msgs geometry_msgs sensor_msgs
  visualization_msgs cv_bridge tf2_ros
)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/calibrate_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
