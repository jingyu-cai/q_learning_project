#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import moveit_commander

# Define colors for the dumbbells
COLOR_BOUNDS = {'red': {'lb': np.array([0, 50, 0]),
                            'ub': np.array([50, 255, 255])},
                    'green': {'lb': np.array([50, 50, 0]),
                            'ub': np.array([100, 255, 255])},
                    'blue': {'lb': np.array([100, 50, 0]), 
                            'ub': np.array([160, 255, 255])}}
COLORS = ['red', 'green', 'blue']

# Define robot statuses to keep track of its actions
GO_TO_DB = "go_to_dumbbell"
REACHED_DB = "reached_db"
PICKED_UP_DB = "picked_up_dumbbell"
MOVING_TO_BLOCK = "moving_to_block"
REACHED_BLOCK = "reached_block"


class RobotPerception(object):
    def __init__(self):

        # Once everything is set up this will be set to true
        self.initialized = False

        # Initialize this node
        rospy.init_node('robot_perception')

        # Set up publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        # Set up subscribers
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # Create an empty Twist msg
        self.twist = Twist()

        # Set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # Initialize array to hold and process images
        self.image = []

        # Initialize array to hold and process scan data
        self.__scan_data = []

        # Minimum distance in front of dumbbell
        self.__goal_dist_in_front_of_dumbbell = 0.2

        # For Sensory-Motor Control in controling the speed
        self.__prop = 0.15 

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # First, robot's status set to GO_TO_DB
        self.robot_status = GO_TO_DB

        # Now everything is initialized
        self.initialized = True
    

    def set_vel(self, diff_ang=0.0, diff_dist=float('inf')):
        """ Set the velocities of the robot """

        ang_v, lin_v = None, None

        # Keep turning if the robot cannot see anything
        if diff_dist == float("inf"):
            print("=====I can't see it! Turning turning=====")
            ang_v = 0.05
            lin_v = 0.0
        
        # Stop if the robot is in front of the dumbbell
        elif diff_dist < self.__goal_dist_in_front_of_dumbbell:
            print("=====I got you.=====")
            ang_v, lin_v = 0.0, 0.0
        
        # Go forwards if robot is still away from dumbbell
        else:
            print("=====Rushing Ahead!=====")
            lin_v = self.__prop * diff_dist
            ang_v = 0.0
        
        return ang_v, lin_v
        

    def pub_vel(self, ang_v=0.0, lin_v=0.0):
        """ To publish a twist to the cmd_vel channel """

        # Set linear and angular velocities and publish
        self.twist.linear.x = lin_v
        self.twist.angular.z = ang_v
        self.cmd_vel_pub.publish(self.twist)


    def scan_callback(self, data):
        """ Store scan data in self.__scan_data to be processed """

        # Store the ranges data
        print("***** Got new scan! *****")
        self.__scan_data = data.ranges


    def move_to_dumbbell(self, color: str):
        """ Move to a dumbbell based on color """

        # Do nothing if there are no images
        if len(self.image) == 0:
            print("-- Have not got the image --")
            return

        # Do nothing if there are no scan data
        if len(self.__scan_data) == 0:
            print("-- Have not got the scan --")
            return

        # Turn image into HSV style
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Get lower and upper bounds of the specified color
        lb, ub = COLOR_BOUNDS[color]['lb'], COLOR_BOUNDS[color]['ub']
        
        # Mask and get moment of the color of the dumbbell
        mask = cv2.inRange(hsv, lb, ub)
        M = cv2.moments(mask)

        # Get the shape of the image to compute its center
        h, w, d = self.image.shape

        # If there are any pixels found for the desired color
        if M['m00'] > 0:

            # Determine the center of the yellow pixels in the image
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # Dumbbell's distance to the center of the camera
            err = w/2 - cx

            print(f"abs(err) / w : {abs(err) / w}")

            # If the color center is at the front
            if abs(err) / w < 0.05:
                
                min_dist = min(self.__scan_data[-10:] + self.__scan_data[:10])

                if min_dist <= self.__goal_dist_in_front_of_dumbbell:

                    # Stop the robot
                    self.pub_vel(0,0)

                    # Sleep 1s to make sure the robot has stopped
                    rospy.sleep(1)
                    self.robot_status = REACHED_DB
                    print(f"---reached dumbbell of color {color}----")

                else:

                    # Rush STRAIGHT toward the dumbbell
                    ang_v, lin_v = self.set_vel(0, min_dist)
                    self.pub_vel(ang_v, lin_v)

            # If the color center is not right at the front yet
            else:
                
                # Define k_p for proportional control            
                k_p = 1.0 / 1000.0

                # Slowly turn the head, so that the color center 
                # would be at the center of the camera
                self.pub_vel(k_p*err, 0)
                print(f"---turning to dumbbell of color {color}----")

        # If we cannot see any pixel of the desired color
        else:

            # Simply turn the head clockwise, without any linear speed
            ang_v, lin_v = self.set_vel()
            self.pub_vel(ang_v, lin_v)

            if self.robot_status == REACHED_DB:
                self.lift_dumbbell()


    def lift_dumbbell(self):
        """ Lift the dumbbell when robot reached the dumbbells """

        # Do nothing if the robot hasn't reached the dumbbells
        if self.robot_status != REACHED_DB:
            return 

        # Set arm and gripper joint goals and move them    
        arm_joint_goal = [0.0, 0.0, -0.45, -0.1]
        gripper_joint_goal = [0.004, 0.004]
        self.move_group_arm(arm_joint_goal, wait=True)
        self.move_group_gripper(gripper_joint_goal, wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()

        # After the robot grapped the dumbbells, now it's time to move to the blocks
        self.robot_status = MOVING_TO_BLOCK


    def image_callback(self, data):
        """ Process the image from the robot's RGB camera """

        # Do nothing if initialization is not done
        if (not self.initialized):
            return

        # Take the ROS message with the image and turn it into a format cv2 can use
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        self.move_to_dumbbell('red')

     
    def run(self):
        """ Run the node """

        # Keep the program alive
        rospy.spin()


if __name__ == "__main__":
    try:
        node = RobotPerception()
        node.run()
    except rospy.ROSInterruptException:
        pass