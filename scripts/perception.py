#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np
import keras_ocr
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import moveit_commander

# Define colors for the dumbbells and block number
COLOR_BOUNDS = {'red': {'lb': np.array([0, 50, 0]),
                            'ub': np.array([50, 255, 255])},
                    'green': {'lb': np.array([50, 50, 0]),
                            'ub': np.array([100, 255, 255])},
                    'blue': {'lb': np.array([100, 50, 0]), 
                            'ub': np.array([160, 255, 255])},
                    'black': {'lb': np.array([0, 0, 0]), 
                            'ub': np.array([180, 255, 50])}}
COLORS = ['red', 'green', 'blue']

# Define robot statuses to keep track of its actions
GO_TO_DB = "go_to_dumbbell"
REACHED_DB = "reached_db"
PICKED_UP_DB = "picked_up_dumbbell"
MOVING_TO_BLOCK = "moving_to_block"
REACHED_BLOCK = "reached_block"

print(f"os cwd: {os.getcwd()}")
# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

# Path of where the trained Q-matrix csv file is located
Q_MATRIX_PATH = "q_matrix.csv"


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

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-9 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][12] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt")

        # Fetch actions. These are the only 9 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { dumbbell: "red", block: 1}
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": COLORS[int(x[0])], "block": int(x[1])},
            self.actions
        ))

        # Set up a list to store the trained Q-matrix
        self.q_matrix = []
        self.load_q_matrix()

        # Set up a list of tuples to store the action sequence
        self.action_sequence = []

        # Create an empty Twist msg
        self.twist = Twist()

        # Set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # Download pre-trained model
        self.pipeline = keras_ocr.pipeline.Pipeline()

        # Initialize array to hold and process images
        self.image = []

        # Initialize array to hold and process scan data
        self.__scan_data = []

        # Minimum distance in front of dumbbell/block
        # ORIGINAL = 0.21
        self.__goal_dist_in_front__db = 0.22
        self.__goal_dist_in_front_block = 0.5

        # For Sensory-Motor Control in controling the speed
        self.__prop = 0.15 

        # The interface to the group of joints making up the turtlebot3
        #   openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # The interface to the group of joints making up the turtlebot3
        #   openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # Initialize starting robot arm and gripper position
        self.initialize_move_group()

        # First, robot's status set to GO_TO_DB
        self.robot_status = PICKED_UP_DB

        # Now everything is initialized
        self.initialized = True

    
    def load_q_matrix(self):
        """ Load the trained Q-matrix csv file """

        # Store the file into self.q_matrix
        self.q_matrix = np.loadtxt(Q_MATRIX_PATH, delimiter = ',')


    def get_action_sequence(self):
        """ Get the sequence of actions for the robot to move the dumbbells
        to the correct blocks based on the trained Q-matrix """

        # Do nothing if initialization is not done
        if not self.initialized:
            return
        
        # Start at the origin
        curr_state = 0

        # Keep track of what dumbbells and blocks are taken
        colors = ["red", "green", "blue"]
        blocks = [1, 2, 3]

        # Loop through 3 times to get the action sequence
        for i in range(3):

            # We can rely on the Q-matrix for the first two actions
            if i != 2:

                # Get row in matrix and select the best action to take
                q_matrix_row = self.q_matrix[curr_state]
                selected_action = np.where(q_matrix_row == max(q_matrix_row))[0][0]

                # Store the dumbbell color and block number for the action as a tuple
                db = self.actions[selected_action]["dumbbell"]
                block = self.actions[selected_action]["block"]
                self.action_sequence.append((db, block))

                # Update current state
                curr_state = np.where(self.action_matrix[curr_state] == selected_action)[0][0]

                # Remove the taken dumbbell and block
                colors.remove(db)
                blocks.remove(block)

            # To avoid invalid actions from the Q-matrix, we eliminate the
            #   first two actions, so the last action will be the third action
            else:
                self.action_sequence.append((colors[0], blocks[0]))

        print(self.action_sequence)


    def set_vel(self, diff_ang=0.0, diff_dist=float('inf')):
        """ Set the velocities of the robot """

        ang_v, lin_v = None, None

        # Keep turning if the robot cannot see anything
        if diff_dist == float("inf"):
            print("=====I can't see it! Turning turning=====")
            ang_v = 0.05
            lin_v = 0.0
        
        # Stop if the robot is in front of the dumbbell
        elif diff_dist < self.__goal_dist_in_front__db:
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


    def initialize_move_group(self):
        """ Initialize the robot arm & gripper position so it can grab onto
        the dumbbell """

        # Set arm and gripper joint goals and move them
        arm_joint_goal = [0.0, 0.65, 0.15, -0.9]
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()


    def lift_dumbbell(self):
        """ Lift the dumbbell when robot reached the dumbbells """

        # Do nothing if the robot hasn't reached the dumbbells
        if self.robot_status != REACHED_DB:
            return 

        # Set arm and gripper joint goals and move them    
        arm_joint_goal = [0.0, 0.05, -0.45, -0.1]
        gripper_joint_goal = [0.004, 0.004]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()

        # Step back
        print("----- stepping back!----")
        self.pub_vel(0, -0.5)
        rospy.sleep(0.8)
        self.pub_vel(0, 0)

        # After the robot grapped the dumbbells, it's time to identify the blocks
        self.robot_status = PICKED_UP_DB


    def drop_dumbbell(self):
        """ Drop the dumbbell when robot reached the blocks """

        # Do nothing is the robot hasn't reached the blocks
        if self.robot_status != REACHED_BLOCK:
            return

        # Set arm and gripper joint goals and move them
        arm_joint_goal = [0.0, 0.45, 0.5, -0.9]
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()

        # After the robot dropped the dumbbells, it's time to go back to the dumbbells
        self.robot_status = GO_TO_DB


    def image_callback(self, data):
        """ Process the image from the robot's RGB camera """

        # Do nothing if initialization is not done
        if (not self.initialized):
            return

        # Take the ROS message with the image and turn it into a format cv2 can use
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')


    def scan_callback(self, data):
        """ Store scan data in self.__scan_data to be processed """

        # Do nothing if initialization is not done
        if (not self.initialized):
            return

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

                print(f"min_dist: {min_dist}")

                if min_dist <= self.__goal_dist_in_front__db:

                    # Stop the robot
                    self.pub_vel(0,0)

                    # Sleep 1s to make sure the robot has stopped
                    rospy.sleep(1)

                    # Change status to reached dumbbell
                    self.robot_status = REACHED_DB
                    print(f"---reached dumbbell of color {color}----")

                    # Lift the dumbbell
                    self.lift_dumbbell()

                else:

                    # Rush STRAIGHT toward the dumbbell
                    ang_v, lin_v = self.set_vel(0, min_dist)
                    self.pub_vel(ang_v, lin_v)

            # If the color center is not right at the front yet
            else:
                
                # Define k_p for proportional control            
                k_p = 1.0 / 1000.0

                # Slowly turn the head, so that the color center 
                #   would be at the center of the camera
                self.pub_vel(k_p * err, 0)
                print(f"---turning to dumbbell of color {color}----")

        # If we cannot see any pixel of the desired color
        else:

            # Simply turn the head clockwise, without any linear speed
            ang_v, lin_v = self.set_vel()
            self.pub_vel(ang_v, lin_v)


    def is_correct_num(self, id: int, prediction_group):
        """ Check if the detected number is the block ID we are looking for """

        # Sometimes the detector may recognize numbers as other numbers or 
        #   characters, so we are grouping them into the same category
        ones = ["1", "l", "i"]
        twos = ["2"]
        threes = ["3", "5", "8", "s", "b", "81"]

        detected_num = 0

        # We always grab the first image in the list for detection, the robot
        #   turning movement will ensure that this will always be the next block
        #   it sees on the left
        if prediction_group[0][0] in ones:
            detected_num = 1
        elif prediction_group[0][0] in twos:
            detected_num = 2
        elif prediction_group[0][0] in threes:
            detected_num = 3

        if detected_num == id:
            return True

        return False

    
    def move_to_block(self, id: int):
        """ Move to a block based on its ID """

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
        lb, ub = COLOR_BOUNDS['black']['lb'], COLOR_BOUNDS['black']['ub']
        
        # Mask and get moment of the color of the dumbbell
        mask = cv2.inRange(hsv, lb, ub)
        M = cv2.moments(mask)

        # If the robot has picked up dumbbell but hasn't detected the desired block yet
        if self.robot_status == PICKED_UP_DB:
            
            # If there are any pixels found for the desired color
            if M['m00'] > 0:
                
                # Stop to look at the block
                self.pub_vel(0, 0)

                # Call the recognizer on the image
                prediction_group = self.pipeline.recognize([self.image])[0]

                # If the recognizer cannot recognize the image, keep turning
                if len(prediction_group) == 0:

                    # Publish a small angular velocity so the robot doesn't overshoot
                    self.pub_vel(0.1, 0)
                    rospy.sleep(0.5)
                    
                # If the recognizer has recognized an image, we check if the first 
                #   one is the correct one
                else:
                    
                    print("Successfully got: " + str(prediction_group[0][0]))

                    # Make sure we have the correct num
                    if self.is_correct_num(id, prediction_group):

                        # Set robot status to move to block
                        self.robot_status = MOVING_TO_BLOCK

                    # Otherwise, we keep turning
                    else:

                        # We will publish a specific degree so that the robot always
                        #   only sees the next block on its left, so we always grab
                        #   the first image in the list for detection
                        self.pub_vel(math.radians(10), 0)
                        rospy.sleep(6)

            # If we cannot see the pixel of the desired color
            else:
                
                # Keep turning
                self.pub_vel(0.3, 0)

        # If the robot has found the desired block, then move to it
        elif self.robot_status == MOVING_TO_BLOCK:

            # TODO: This needs to be revised??? 
            # I only checked when robot moves to block number 2 and it works, not sure about 1 and 3

            # Get the shape of the image to compute its center
            h, w, d = self.image.shape

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # Dumbbell's distance to the center of the camera
            err = w/2 - cx

            print(f"abs(err) / w : {abs(err) / w}")

            # If the color center is at the front
            if abs(err) / w < 0.05:
                
                min_dist = min(self.__scan_data[-10:] + self.__scan_data[:10])

                print(f"min_dist: {min_dist}")

                if min_dist <= self.__goal_dist_in_front_block:

                    # Stop the robot
                    self.pub_vel(0,0)

                    # Sleep 1s to make sure the robot has stopped
                    rospy.sleep(1)

                    # Change status to reached block
                    self.robot_status = REACHED_BLOCK
                    print(f"---reached block of ID {id}----")

                    # Drop the dumbbell
                    self.drop_dumbbell()

                else:

                    # Rush STRAIGHT toward the block
                    ang_v, lin_v = self.set_vel(0, min_dist)
                    self.pub_vel(ang_v, lin_v)

            # If the color center is not right at the front yet
            else:
                
                # Define k_p for proportional control            
                k_p = 1.0 / 1000.0

                # Slowly turn the head, so that the color center 
                #   would be at the center of the camera
                self.pub_vel(k_p * err, 0)
                print(f"---turning to block of ID {id}----")

        # Do nothing if the robot isn't in any of the two statuses
        else:
            return
        

    def run(self):
        """ Run the node """

        # Set a rate to maintain the frequency of execution
        r = rospy.Rate(5)

        # Run the program based on different statuses
        # TODO: This part needs to be automated
        while not rospy.is_shutdown():
            if self.robot_status == GO_TO_DB:
                self.move_to_dumbbell('red')
            elif self.robot_status == PICKED_UP_DB or self.robot_status == MOVING_TO_BLOCK:
                self.move_to_block(1)
            
            r.sleep()


if __name__ == "__main__":
    try:
        node = RobotPerception()
        node.run()
    except rospy.ROSInterruptException:
        pass