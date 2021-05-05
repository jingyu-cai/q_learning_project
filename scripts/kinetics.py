#!/usr/bin/env python3

import rospy
# import the moveit_commander, which allows us to control the arms
import moveit_commander
import math
# import the custom message

class TestRobot(object):

    def __init__(self):
        rospy.init_node('turtlebot3_test')
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.move_group_arm.go([0,0,0,0], wait=True)
        print("ready")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    robot = TestRobot()
    robot.run()

    '''
    Gives Error: 
    Traceback (most recent call last):
    File "/home/zhouxing/catkin_ws/src/q_learning_project/scripts/kinetics.py", line 22, in <module>
        robot = TestRobot()
    File "/home/zhouxing/catkin_ws/src/q_learning_project/scripts/kinetics.py", line 13, in __init__
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
    File "/opt/ros/noetic/lib/python3/dist-packages/moveit_commander/move_group.py", line 53, in __init__
        self._g = _moveit_move_group_interface.MoveGroupInterface(name, robot_description, ns, wait_for_servers)
    RuntimeError: Group 'arm' was not found.
    '''