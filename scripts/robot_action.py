#!/usr/bin/env python3

import rospy, rospkg, cv2, cv_bridge
import os
import numpy as np
import keras_ocr
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

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
MEASURE_ANGLE = "measure_angle"
GO_TO_DB = "go_to_dumbbell"
REACHED_DB = "reached_db"
PICKED_UP_DB = "picked_up_dumbbell"
MOVING_TO_BLOCK = "moving_to_block"
REACHED_BLOCK = "reached_block"

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

# Path of where the trained Q-matrix csv file is located
q_matrix_path = os.path.dirname(__file__) + "/q_matrix.csv"


class RobotAction(object):


    def __init__(self):

        # Once everything is set up this will be set to true
        self.initialized = False

        # Initialize this node
        rospy.init_node('robot_action')

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

        # Set up a list of tuples to store the action sequence and populate it
        #   using the converged Q-matrix
        self.action_sequence = []
        self.get_action_sequence()

        # Initialize number to keep track which action step we are on
        self.action_step = 0

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

        # The "inf" in the scan data
        self.scan_max = None

        # Initialize the ranges for blocks and void        
        self.inf_bounds = [[0, 0] for _ in range(4)]
        self.inf_bounds[-1][1] = 90
        self.block_bounds = [[0, 0] for _ in range(3)]

        # Initialize the large_angle to be used in turning
        self.large_angle = -1

        # Minimum distance in front of dumbbell and block
        self.__goal_dist_in_front__db = 0.22
        self.__goal_dist_in_front_block = 0.55

        # For sensory-motor Ccntrol in controlling the speed
        self.__prop = 0.15 

        # The interface to the group of joints making up the turtlebot3
        #   openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")

        # The interface to the group of joints making up the turtlebot3
        #   openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # Initialize starting robot arm and gripper positions
        self.initialize_move_group()

        # First, robot's status set to MEASURE_ANGLE to measure the angles
        #   between the blocks
        self.robot_status = MEASURE_ANGLE

        # Now everything is initialized
        self.initialized = True

    
    def load_q_matrix(self):
        """ Load the trained Q-matrix csv file """

        # Store the file into self.q_matrix
        self.q_matrix = np.loadtxt(q_matrix_path, delimiter = ',')


    def get_action_sequence(self):
        """ Get the sequence of actions for the robot to move the dumbbells
        to the correct blocks based on the trained Q-matrix """
        
        # Start at the origin
        curr_state = 0

        # Loop through 3 times to get the action sequence
        for i in range(3):

            # Get row in matrix and select the best action to take with the
            #   maximum Q-value
            q_matrix_row = self.q_matrix[curr_state]
            selected_action = np.where(q_matrix_row == max(q_matrix_row))[0][0]

            # Store the dumbbell color and block number for the action as a tuple
            db = self.actions[selected_action]["dumbbell"]
            block = self.actions[selected_action]["block"]
            self.action_sequence.append((db, block))

            # Update the current state
            curr_state = np.where(self.action_matrix[curr_state] == selected_action)[0][0]
                
        print(self.action_sequence)


    def set_vel(self, diff_ang=0.0, diff_dist=float('inf')):
        """ Set the velocities of the robot """

        ang_v, lin_v = None, None

        # Keep turning if the robot cannot see anything
        if diff_dist == float("inf"):
            print("=====I can't see it! Turning turning=====")
            ang_v = 0.15
            lin_v = 0.0
        
        # Stop if the robot is in front of the dumbbell/block
        elif diff_dist < self.__goal_dist_in_front__db:
            print("=====Arrived!=====")
            ang_v, lin_v = 0.0, 0.0
        
        # Go forwards if robot is still away from dumbbell/block
        else:
            print("=====Rushing Ahead!=====")
            lin_v = self.__prop * diff_dist
            ang_v = 0.0
        
        return ang_v, lin_v
        

    def pub_vel(self, ang_v=0.0, lin_v=0.0):
        """ Publish a twist to the cmd_vel channel """

        # Set linear and angular velocities and publish
        self.twist.linear.x = lin_v
        self.twist.angular.z = ang_v
        self.cmd_vel_pub.publish(self.twist)


    def compute_large_angle(self, block_bounds):
        """ Compute the large angle between blocks for turning, this would
        allow the robot to turn from facing one block to facing the next one """

        # Initialize three midpoints of the three blocks
        midpoints = [0, 0, 0]

        # For each block bound, we compute its midpoint with some added bias
        for i in range(len(block_bounds)):
            
            # Lower and upper bound angles for the block
            lb, ub = block_bounds[i]

            if i == 0:
                # Add some bias for the rightmost block
                midpoints[i] = lb - 9

            elif i == 1:
                # Just take the midpoint is fine for the middle block
                midpoints[i] = (lb + ub) / 2

            elif i == 2:
                # Add some bias for the leftmost block
                midpoints[i] = ub + 9
        
        # Subtracting each midpoint to approximate the angle for the robot to turn
        diffs = [midpoints[1] - midpoints[0], midpoints[2] - midpoints[1]]
        
        return math.radians(sum(diffs) / 2)  


    def process_scan(self):
        """ Process the scan data to get the large angle """

        # Do nothing if the status is not measure_angle
        if self.robot_status != MEASURE_ANGLE:
            return

        # Do nothing if the large_angle has already been set
        if self.large_angle != -1:
            return

        # Do nothing if initialization is not complete
        if not self.initialized:
            return

        # Do nothing if no scan data is received
        if len(self.__scan_data) == 0:
            print("Have not got the __scan_data yet")
            return
        
        # Initially, the robot faces to the dumbbells. To get the angles for 
        #   blocks, we use the scan data from the back. Note that the scan data
        #   is a list of length 360, with 0 -> front, 90 -> right, 180 -> back,
        #   270 -> left

        # In this data range, we expect four consecutive parts with value inf
        #   and hence three consecutive parts with value != inf. We save the 
        #   upper bound and lower bound of these ranges in self.inf_bounds and 
        #   self.block_bounds

        cnt = 0
        
        # A flag for marking the start and end of an "inf" range
        seen = False

        for idx in range(90, 270):

            if self.__scan_data[idx] < self.scan_max:

                # If we have marked the start of the inf range, when we 
                #   encounter a value != inf, it means that this is the end
                #   of current inf range
                if seen == True:
                    self.inf_bounds[cnt][1] = idx
                    seen = False
                    cnt += 1
            else:

                # If the previous inf range has ended, and we encounter a
                #   new inf value, we take it as the start of an inf range
                if not seen:
                    self.inf_bounds[cnt][0] = idx
                    seen = True       

        # There are 4 inf ranges, and the block ranges are just intervals between
        #   these inf ranges
        for i in range(3):
            self.block_bounds[i][0] = self.inf_bounds[i][1]
            self.block_bounds[i][1] = self.inf_bounds[i+1][0]

        # For testing, we print out the block ranges 
        print(f"self.block_bounds = {self.block_bounds}")  
        self.large_angle = self.compute_large_angle(self.block_bounds)
        print(f"self.large_angle = {self.large_angle}")

        # Now that we know the large_angle, the robot can start the actions
        self.robot_status = GO_TO_DB


    def initialize_move_group(self):
        """ Initialize the robot arm & gripper position so it can grab onto
        the dumbbell """

        # Set arm and gripper joint goals and move them
        arm_joint_goal = [0.0, 0.65, 0.15, -0.9]
        gripper_joint_goal = [0.015, 0.015]
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

        # Step back so the robot won't hit the dumbbell while rotating
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
        arm_joint_goal = [0.0, 0.65, 0.15, -0.9]
        gripper_joint_goal = [0.015, 0.015]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        rospy.sleep(0.5)
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.sleep(1)

        # Step back so the robot won't hit the dumbbell while rotating
        print("----- stepping back!----")
        self.pub_vel(0, -0.3)
        rospy.sleep(1)
        self.pub_vel(0, 0)

        # After the robot dropped the dumbbells, it's time to go back to the dumbbells
        self.robot_status = GO_TO_DB

        # We also increase the number of action steps by 1
        self.action_step += 1


    def image_callback(self, data):
        """ Process the image from the robot's RGB camera """

        # Do nothing if initialization is not done
        if not self.initialized:
            return

        # Take the ROS message with the image and turn it into a format cv2 can use
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')


    def scan_callback(self, data):
        """ Store scan data in self.__scan_data to be processed """

        # Do nothing if initialization is not done
        if not self.initialized:
            return

        # Store the ranges data
        self.__scan_data = data.ranges

        if not self.scan_max:
            self.scan_max = data.range_max


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

            # Determine the center of the colored pixels in the image
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # Dumbbell's distance to the center of the camera
            err = w/2 - cx

            print(f"abs(err) / w : {abs(err) / w}")

            # If the color center is at the front
            if abs(err) / w < 0.05:
                
                min_dist = min(self.__scan_data[-10:] + self.__scan_data[:10])

                print(f"min_dist: {min_dist}")

                # If the robot is close to the dumbbell
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


    def leftmost_is_correct_id(self, id: int, prediction_group):
        """ Check if the detected number on the leftmost block in the camera view
        is the block ID we are looking for; the robot turning movement with 
        self.large_angle will ensure that the robot will always face the next 
        block on the left """

        # Define variables to be used in the comparison step below
        left_most_block = 0
        center_x = sum([coord[0] for coord in prediction_group[left_most_block][1]]) / 4
        
        # Find the left-most image using the center of the x positions of the boxes
        for i in range(len(prediction_group)):

            # Compute the center of the x positions
            temp_center_x = sum([coord[0] for coord in prediction_group[i][1]]) / 4

            # Update left_most_block and center_x if the new center is smaller / more left
            if temp_center_x < center_x:
                left_most_block = i
                center_x = temp_center_x

        print("Successfully got: " + str(prediction_group[left_most_block][0]))

        # Sometimes the detector may recognize numbers as other numbers or 
        #   characters, so we are grouping them into the same category
        ones = ["1", "l", "i"]
        twos = ["2"]
        threes = ["3", "5", "8", "s", "b", "81"]

        detected_num = 0

        # We grab the leftmost image in the list for detection and grouping
        if prediction_group[left_most_block][0] in ones:
            detected_num = 1
        elif prediction_group[left_most_block][0] in twos:
            detected_num = 2
        elif prediction_group[left_most_block][0] in threes:
            detected_num = 3

        # Return true if the detected number matches our desired ID
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
        
        # Mask and get moment of the color of the block
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
                    self.pub_vel(0.05, 0)
                    rospy.sleep(0.5)
                    
                # If the recognizer has recognized an image, we check if the 
                #   leftmost one is the correct one
                else:

                    # Make sure we have the correct num for the leftmost block
                    if self.leftmost_is_correct_id(id, prediction_group):

                        # Set robot status to move to block
                        self.robot_status = MOVING_TO_BLOCK

                    # Otherwise, we keep turning
                    else:

                        # We will publish the computed angle between the blocks so 
                        #   that the robot always only sees the next block on its left
                        spin_speed = math.radians(9.5)
                        self.pub_vel(spin_speed, 0)
                        rospy.sleep(self.large_angle / spin_speed)

            # If we cannot see the pixel of the desired color
            else:
                
                # Keep turning
                self.pub_vel(0.3, 0)

        # If the robot has found the desired block, then move to it
        elif self.robot_status == MOVING_TO_BLOCK:

            # Get the shape of the image to compute its center
            h, w, d = self.image.shape

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # Block number's distance to the center of the camera
            err = w/2 - cx

            print(f"abs(err) / w : {abs(err) / w}")
                
            min_dist = min(self.__scan_data[-10:] + self.__scan_data[:10])

            print(f"min_dist: {min_dist}")

            # If the robot is close to the block
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
                    
                # Define k_p for proportional control            
                k_p = 1.0 / 2500.0

                # Slowly turn the head while going forwards, so that the 
                #   color center would be at the center of the camera
                self.pub_vel(k_p * err, 0.3)
                print(f"---turning to block of ID {id}----")

        # Do nothing if the robot isn't in any of the two statuses
        else:
            return
        

    def run(self):
        """ Run the node """

        # Set a rate to maintain the frequency of execution
        r = rospy.Rate(5)

        # Run the program based on different statuses and number of action steps
        while not rospy.is_shutdown():
            
            # Process the scan data first to measure the angles between the blocks
            if self.robot_status == MEASURE_ANGLE:
                self.process_scan()

            # If we haven't exhausted the action sequence list yet, then we keep taking actions
            if self.action_step < 3:

                # If the robot is in these two statuses, then it needs to execute move_to_dumbbell
                if self.robot_status == GO_TO_DB or self.robot_status == REACHED_DB:
                    self.move_to_dumbbell(self.action_sequence[self.action_step][0])

                # If the robot is in these three statuses, then it needs to execute move_to_block
                elif self.robot_status == PICKED_UP_DB or self.robot_status == MOVING_TO_BLOCK or self.robot_status == REACHED_BLOCK:
                    self.move_to_block(self.action_sequence[self.action_step][1])
            
            r.sleep()


if __name__ == "__main__":

    # Declare a node and run it
    try:
        node = RobotAction()
        node.run()
    except rospy.ROSInterruptException:
        pass