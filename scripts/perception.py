#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import os

COLOR_BOUNDS = {'red': {'lb': np.array([0, 50, 0]),
                            'ub': np.array([50, 255, 255])},
                    'green': {'lb': np.array([50, 50, 0]),
                            'ub': np.array([100, 255, 255])},
                    'blue': {'lb': np.array([100, 50, 0]), 
                            'ub': np.array([160, 255, 255])}}
COLORS = ['red', 'green', 'blue']

class DigitRecognizer(object):
    def __init__(self):
        self.initalized = False
        rospy.init_node('digit_recognizer')


        self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
        Image, self.image_callback)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel',
                        Twist, queue_size=1)

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        self.bridge = cv_bridge.CvBridge()


        self.image = []

        self.__scan_data = []

        self.__goal_dist_in_front_of_dumbell = 0.35

        self.__prop = 0.15

        self.initalized = True
    
    def set_vel(self, diff_ang=0.0, diff_dist=float('inf')):

        ang_v, lin_v = None, None
        if diff_dist == float("inf"):
            print("=====I can't see it! Turning turning=====")
            ang_v = 0.05
            lin_v = 0.0
        
        elif diff_dist < self.__goal_dist_in_front_of_dumbell:
            print("=====I got you.=====")
            ang_v, lin_v = 0.0, 0.0
        
        else:
            print("=====Rushing Ahead!=====")
            lin_v = self.__prop * diff_dist
            ang_v = 0.0
        
        return ang_v, lin_v
        

    def pub_vel(self, ang_v=0.0, lin_v=0.0):
        new_twist = Twist()
        new_twist.linear.x = lin_v
        new_twist.angular.z = ang_v
        self.cmd_vel_pub.publish(new_twist)

    def scan_callback(self, data):
        print("***** Got new scan! *****")
        self.__scan_data = data.ranges

    def move_to_dumbell(self, color: str):
        if len(self.image) == 0:
            print("-- Have not got the image --")
            return
        if len(self.__scan_data) == 0:
            print("-- Have not got the scan --")
            return

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        lb, ub = COLOR_BOUNDS[color]['lb'], COLOR_BOUNDS[color]['ub']
        
        mask = cv2.inRange(hsv, lb, ub)

        h, w, d = self.image.shape

        M = cv2.moments(mask)

        # if there are any yellow pixels found
        if M['m00'] > 0:
            # determine the center of the yellow pixels in the image
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            err = w/2 - cx

            cv2.circle(self.image, (cx, cy), 20, (255,155,0), -1)

            print(f"abs(err) / w : {abs(err) / w}")

            if abs(err) / w < 0.05:
                # The color is at the front
                min_dist = min(self.__scan_data[-10:] + self.__scan_data[:10])
                if min_dist <= self.__goal_dist_in_front_of_dumbell:
                    self.pub_vel(0,0)
                    print(f"---reached dumbell of color {color}----")
                else:
                    ang_v, lin_v = self.set_vel(0, min_dist)
                    self.pub_vel(ang_v, lin_v)
            else:            
                # visualize a red circle in our debugging window to indicate
                # the center point of the yellow pixels
                k_p = 1.0 / 1000.0
                self.pub_vel(k_p*err, 0)
                print(f"---turning to dumbell of color {color}----")
        else:
            ang_v, lin_v = self.set_vel()
            self.pub_vel(ang_v, lin_v)

            


    def image_callback(self, data):
        if (not self.initalized):
            return


            # take the ROS message with the image and turn it into a format cv2 can use
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.move_to_dumbell('red')
            
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = DigitRecognizer()
        node.run()
    except rospy.ROSInterruptException:
        pass