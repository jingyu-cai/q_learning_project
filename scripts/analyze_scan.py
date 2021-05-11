#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np
import math

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import utils

class AnalyzeScan(object):
    def __init__(self):

        self.initialized = False
        rospy.init_node('analyzeScan')
        # Set up publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
    
        self.scan_data = []

        self.scan_max = None

        self.inf_bounds = [[0, 0] for _ in range(4)]
        self.inf_bounds[-1][1] = 90

        self.block_bounds = [[0, 0] for _ in range(3)]

        self.large_angle = -1

        self.turn_to_three = False

        rospy.sleep(2)

        self.initialized = True
    
    def scan_callback(self, data):
        if not self.initialized:
            return
        self.scan_data = data.ranges
        if not self.scan_max:
            self.scan_max = data.range_max
        
        # print(self.scan_data[-90:] + self.scan_data[:90])
        # print("============")
        self.process_scan()
        #print(f"self.inf_bounds = {self.inf_bounds}")
        rospy.sleep(10)
    
    def process_scan(self):
        if self.large_angle != -1:
            return

        if self.initialized == False:
            return False
        if len(self.scan_data) == 0:
            print("Have not got the scan_data yet")
            return False
        
        
        cnt = 0
        seen = False
        for idx in range(-90, 90):
            if self.scan_data[idx] < self.scan_max:
                if seen == True:
                    self.inf_bounds[cnt][1] = idx
                    seen = False
                    cnt += 1
            else:
                if not seen:
                    self.inf_bounds[cnt][0] = idx
                    seen = True       

        for i in range(3):
            self.block_bounds[i][0] = self.inf_bounds[i][1]
            self.block_bounds[i][1] = self.inf_bounds[i+1][0]

        print(f"self.block_bounds = {self.block_bounds}")  
        self.large_angle = self.compute_large_angle()
        print(f"self.large_angle = {self.large_angle}")

        new_twist = Twist()
        new_twist.angular.z = -math.radians(12)
        self.cmd_vel_pub.publish(new_twist)
        rospy.sleep(6)

        print("---- facing to 3 ----")

        self.stop_robot()
        self.turn_to_three = True


    def compute_large_angle(self):
        midpoints = [0, 0, 0]
        for i in range(len(self.block_bounds)):
            lb, ub = self.block_bounds[i]
            if i == 0:
                midpoints[i] = lb - 14 # Add some bias
            elif i == 1:
                midpoints[i] = (3 * lb + 3 * ub) / 6
            elif i == 2:
                midpoints[i] = ub + 14 # Add some bias
            
        
        diffs = [midpoints[1] - midpoints[0], midpoints[2] - midpoints[1]]
        return sum(diffs) / 2       

    def stop_robot(self):
        pause_twist = Twist()
        pause_twist.angular.z = 0.0
        pause_twist.linear.x = 0.0
        pause_twist.linear.y = 0.0
        self.cmd_vel_pub.publish(pause_twist)
        rospy.sleep(5)

    def rotate_robot(self):
        if self.large_angle == -1 or not self.turn_to_three:
            return
        
        default_spin_speed = 0.5

        new_twist = Twist()
        new_twist.angular.z = default_spin_speed
        spin_time = math.radians(self.large_angle) / default_spin_speed
        self.cmd_vel_pub.publish(new_twist)
        rospy.sleep(spin_time)
        
        self.stop_robot()
        print(" === Turned one big angle === ")

        


    def run(self):
        """ Run the node """

        # Set a rate to maintain the frequency of execution
        r = rospy.Rate(5)

        while not rospy.is_shutdown():
            self.rotate_robot()
            r.sleep()
    
    

if __name__ == "__main__":
    try:
        node = AnalyzeScan()
        node.run()
    except rospy.ROSInterruptException:
        pass