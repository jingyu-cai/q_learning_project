#!/usr/bin/env python3

import rospy
import os
import numpy as np
from numpy.random import choice

from q_learning_project.msg import QLearningReward, QMatrix, QMatrixRow, RobotMoveDBToBlock


# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

print_header = "=" * 10


def convert_q_matrix_to_list(qmatrix):
    """ Helper function to convert Q-matrix to list """

    res = []

    for qrow in qmatrix:
        res.append(qrow.q_matrix_row)

    return res


def print_state(states):
    """ Helper function to print states """

    colors = ['red', 'green', 'blue']
    blocks = states
    output = ""

    for c, b in zip(colors, blocks):
        output += f"{c} -> {b} ;"

    print(output)


class QLearning(object):


    def __init__(self):

        # Once everything is set up this will be set to true
        self.initialized = False

        # Initialize this node
        rospy.init_node("q_learning")

        # Set up publishers
        self.q_matrix_pub = rospy.Publisher("/q_learning/q_matrix", QMatrix, queue_size = 10)
        self.robot_action_pub = rospy.Publisher("/q_learning/robot_action", RobotMoveDBToBlock, queue_size = 10)
        
        self.cnt = 0

        # Set up subscriber
        rospy.Subscriber("/q_learning/reward", QLearningReward, self.reward_received)

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
        colors = ["red", "green", "blue"]
        self.actions = np.loadtxt(path_prefix + "actions.txt")
        self.actions = list(map(
            lambda x: {"dumbbell": colors[int(x[0])], "block": int(x[1])},
            self.actions
        ))

        # Fetch states. There are 64 states. Each row index corresponds to the
        # state number, and the value is a list of 3 items indicating the positions
        # of the red, green, blue dumbbells respectively.
        # e.g. [[0, 0, 0], [1, 0 , 0], [2, 0, 0], ..., [3, 3, 3]]
        # e.g. [0, 1, 2] indicates that the green dumbbell is at block 1, and blue at block 2.
        # A value of 0 corresponds to the origin. 1/2/3 corresponds to the block number.
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.txt")
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        # Initialize current state and keep track of the next state
        self.curr_state = 0
        self.next_state = 0

        # Initialize current action index
        self.curr_action = 0

        # Initialize variables to define static status and keep track of how many 
        #   iterations have the Q-matrix remained static
        self.epsilon = 1
        self.static_iter_threshold = 500
        self.static_tracker = 0

        # Initialize and publish Q-matrix
        self.q_matrix = QMatrix()
        self.initialize_q_matrix()
        self.q_matrix_pub.publish(self.q_matrix)

        # Now everything is initialized, sleep for 1 second to make sure
        self.initialized = True
        rospy.sleep(1)
        
        # Start with a random action
        self.select_random_action()


    def initialize_q_matrix(self):
        """ Initialize the Q-matrix with all 0s to start """

        # Loop over 64 rows and 9 columns to set up the matrix
        for i in range(len(self.states)):
            q_matrix_row = QMatrixRow()
            for j in range(len(self.actions)):
                q_matrix_row.q_matrix_row.append(0)
            self.q_matrix.q_matrix.append(q_matrix_row)


    def select_random_action(self):
        """ Select a random action based on current state and publish it """

        self.cnt += 1

        # Do nothing if Q-matrix is not yet initialized
        if not self.initialized:
            print(print_header + "not initialized" + print_header)
            return
        
        # Identify current state and find the row corresponding with that state in the 
        #   action matrix, this is all the valid + invalid actions a robot can take
        curr_state = self.curr_state
        actions_in_row = self.action_matrix[curr_state]

        # Filter out the invalid actions from the row of actions
        filtered_actions_in_row = list(filter(lambda x: x != -1, actions_in_row))

        # If there are no possible actions to take: reset current state to state 0
        while len(filtered_actions_in_row) == 0:
            print(print_header + "no action to take" + print_header)
            self.curr_state = 0
            curr_state = self.curr_state
            actions_in_row = self.action_matrix[curr_state]
            filtered_actions_in_row = list(filter(lambda x: x != -1, actions_in_row))
            
        # Randomly select an action from the row, assign that action to self.action
        #   and find its index in the row to assign it to self.next_state
        selected_action = int(choice(filtered_actions_in_row))
        self.curr_action = selected_action
        self.next_state = np.where(actions_in_row == selected_action)[0][0]

        # Get the dumbbell color and the block id for the selected action
        db = self.actions[selected_action]["dumbbell"]
        block = self.actions[selected_action]["block"]

        # Set up a RobotMoveDBToBlock() msg and publish it
        robot_action = RobotMoveDBToBlock()
        robot_action.robot_db = db
        robot_action.block_id = block
        self.robot_action_pub.publish(robot_action)
        print(print_header + f"[{self.cnt}] published a new action: {db}, {block}" + print_header)


    def update_q_matrix(self, reward):
        """ Apply the Q-learning algorithm to update and publish the Q-matrix """

        # Initialize variables to be used
        curr_state = self.curr_state
        next_state = self.next_state
        curr_action = self.curr_action

        # Set up parameters for the algorithm
        alpha = 1
        gamma = 0.5

        # Apply algorithm to update the q value for a state-action pair
        old_q_value = self.q_matrix.q_matrix[curr_state].q_matrix_row[curr_action]
        new_q_value = old_q_value + int(alpha * (reward + gamma * max(self.q_matrix.q_matrix[next_state].q_matrix_row) - old_q_value))
        self.q_matrix.q_matrix[curr_state].q_matrix_row[curr_action] = new_q_value

        # Now, move the current state on to the next state
        self.curr_state = next_state

        # For testing: print current state
        print_state(self.states[self.curr_state])

        # Check if the change in q-value is static or not and update the tracker
        if abs(old_q_value - new_q_value) <= self.epsilon:
            self.static_tracker += 1
        else:
            self.static_tracker = 0

        # Publish the Q-matrix
        self.q_matrix_pub.publish(self.q_matrix)


    def is_converged(self):
        """ Check if the Q-matrix has converged """

        # If the Q-matrix has remained static for a certain amount of time, 
        #   then it is defined to be convergent
        if self.static_tracker >= self.static_iter_threshold:
            return True

        return False


    def save_q_matrix(self):
        """ Save Q-matrix as a csv file once it's converged to avoid retraining """

        # Save the Q-matrix as a csv file
        data = self.q_matrix.q_matrix
        data = convert_q_matrix_to_list(data)
        print(f"type of data[0]: {type(data[0])}")
        data = np.asarray(data)

        np.savetxt(os.path.dirname(__file__) + "/q_matrix.csv", data, fmt='%5s', delimiter = ',')

    
    def reward_received(self, data):
        """ Process received reward after an action """

        # Update the Q-matrix
        self.update_q_matrix(data.reward)

        if self.is_converged():
            # If the Q-matrix has converged, then we will save it
            self.save_q_matrix()
            print(print_header + f"matrix saved after {self.cnt} actions!" + print_header)
            exit()
        else:
            # If not, we continue to make random actions
            self.select_random_action()


    def run(self):
        """ Run the node """

        # Keep the program alive
        rospy.spin()


if __name__ == "__main__":

    # Declare a node and run it
    node = QLearning()
    node.run()
