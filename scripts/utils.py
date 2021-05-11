#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np
import math

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

def stop_robot(vel_channel):
    new_twist = Twist()
    new_twist.angular.z = 0.0
    new_twist.linear.x = 0.0
    new_twist.linear.y = 0.0
    vel_channel.publish(new_twist)
    print("--- stopped ---")
    return

def turn_a_pi(vel_channel):
    new_twist = Twist()
    new_twist.angular.z = math.radians(36)
    vel_channel.publish(new_twist)
    stop_robot(vel_channel)
    print('--- Turned a pi ---')
    return



def compute_large_angle(block_bounds):
    midpoints = [0, 0, 0]
    for i in range(len(block_bounds)):
        lb, ub = block_bounds[i]
        if i == 0:
            midpoints[i] = lb - 9.5 # Add some bias
        elif i == 1:
            midpoints[i] = (lb + ub) / 2
        elif i == 2:
            midpoints[i] = ub + 9.5 # Add some bias
        
    
    diffs = [midpoints[1] - midpoints[0], midpoints[2] - midpoints[1]]
    return math.radians(sum(diffs) / 2)  


class BackToOrigin(object):
    def __init__(self, vel_channel):
        self.vel_channel = vel_channel  
        self.liner_dist = 0.0
        self.cur_yaw = 0.0
        self.theta = 0.0

        self.default_spin_speed = 0.2
        self.default_forward_speed = 0.1

        print("---- Init backtoorigin ----")
        

    
    def run(self, current_pos):
        # The pipeline to move robot

        print("==== running back to origin ====")

        cur_x = current_pos.position.x
        cur_y = current_pos.position.y
        
        self.liner_dist = np.sqrt(cur_x ** 2 + cur_y ** 2)
        self.cur_yaw = get_yaw_from_pose(current_pos)
        self.theta = math.atan(cur_x / cur_y)
        

        self.rotate_to_origin()
        self.rush_to_origin()
        self.face_to_right()
    
    
    def stop_robot(self):
        pause_twist = Twist()
        pause_twist.angular.z = 0.0
        pause_twist.linear.x = 0.0
        pause_twist.linear.y = 0.0
        self.vel_channel.publish(pause_twist)
        rospy.sleep(5)

    def rotate_to_origin(self):
        diff_counter_clockwise = math.pi - self.cur_yaw + self.theta
        print(f" ***** Counterclockwise for {np.degrees(diff_counter_clockwise)} degrees")
        print(f"Linear dist = {self.liner_dist}")

        new_twist = Twist()
        new_twist.angular.z = np.sign(diff_counter_clockwise) * self.default_spin_speed
        self.vel_channel.publish(new_twist)
        rospy.sleep(abs(diff_counter_clockwise) / self.default_spin_speed)

        self.stop_robot()

        print("==== rotated_to_origin ===")

    def rush_to_origin(self):

        rush_twist = Twist()
        rush_twist.linear.x = self.default_forward_speed
        rush_twist.angular.z = 0.0
        self.vel_channel.publish(rush_twist)
        rospy.sleep(self.liner_dist / self.default_forward_speed)

        self.stop_robot()

        print("==== rushed_to_origin ===")


    def face_to_right(self):
        '''When have moved the origin, rotate the robot to make it face to the right'''
        straight_twist = Twist()
        straight_twist.angular.z = (-1) * np.sign(self.theta) * self.default_spin_speed
        self.vel_channel.publish(straight_twist)
        rospy.sleep(abs(self.theta) / self.default_spin_speed)

        self.stop_robot()

        right_twist = Twist()
        right_twist.angular.z = - math.radians(15)
        self.vel_channel.publish(right_twist)
        rospy.sleep(6)

        self.stop_robot()
        print("=== facing right now ===")


