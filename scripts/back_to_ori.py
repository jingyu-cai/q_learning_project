#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np
import math


from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import utils

class Move2Origin(object):
    def __init__(self):

        self.initialized = False
        rospy.init_node('move_to_origin')

        # Set up publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.odem_sub = rospy.Subscriber('/odom', Odometry, self.odem_callback)
        
        rospy.sleep(2)
        self.current_pos = []
        self.scan_data = []

        self.initialized = True

    
    def odem_callback(self, msg):
        if self.initialized == False:
            return
        self.current_pos = msg.pose.pose
        print("===============")
        print(f"self.current_pos: {self.current_pos}")
        print(f"get yaw: {utils.get_yaw_from_pose(self.current_pos)}")
        print("===============")
        cur_x = self.current_pos.position.x
        cur_y = self.current_pos.position.y
        liner_dist = np.sqrt(cur_x ** 2 + cur_y ** 2)
        cur_yaw = utils.get_yaw_from_pose(self.current_pos)
        theta = math.atan(cur_x / cur_y)
        diff_counter_clockwise = math.pi - cur_yaw + theta
        print(f" ***** Counterclockwise for {np.degrees(diff_counter_clockwise)} degrees")
        print(f"Linear dist = {liner_dist}")
        
        default_spin_speed = 0.2
        default_forward_speed = 0.1

        new_twist = Twist()
        new_twist.angular.z = np.sign(diff_counter_clockwise) * default_spin_speed
        self.cmd_vel_pub.publish(new_twist)
        rospy.sleep(abs(diff_counter_clockwise) / default_spin_speed)
        
        pause_twist = Twist()
        pause_twist.angular.z = 0.0
        pause_twist.linear.x = 0.0
        pause_twist.linear.y = 0.0
        self.cmd_vel_pub.publish(pause_twist)
        rospy.sleep(1)

        print("==== turned ===")

        rush_twist = Twist()
        rush_twist.linear.x = default_forward_speed
        rush_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(rush_twist)
        rospy.sleep(liner_dist / default_forward_speed)

        self.cmd_vel_pub.publish(pause_twist)
        rospy.sleep(1)
        print("==== rushed ====")

        straight_twist = Twist()
        straight_twist.angular.z = (-1) * np.sign(theta) * default_spin_speed
        self.cmd_vel_pub.publish(straight_twist)
        rospy.sleep(abs(theta) / default_spin_speed)

        self.cmd_vel_pub.publish(pause_twist)
        rospy.sleep(1)
        print("==== straighten ===")

        
        rospy.sleep(10)

    def scan_callback(self, data):
        self.scan_data = data.ranges
        #print(self.scan_data)
        rospy.sleep(5)
    
    def run(self):
        """ Run the node """

        # Set a rate to maintain the frequency of execution
        r = rospy.Rate(5)

        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    try:
        node = Move2Origin()
        node.run()
    except rospy.ROSInterruptException:
        pass