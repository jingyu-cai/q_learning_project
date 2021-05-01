#!/usr/bin/env python3

import rospy
import os

import numpy as np
from numpy.random import choice

from q_learning_project.msg import QLearningReward, QMatrix, QMatrixRow, RobotMoveDBToBlock


# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"


class QLearning(object):


    def __init__(self):
        # Once everything is set up this will be set to true
        self.initialized = False

        # Initialize this node
        rospy.init_node("q_learning")

        # Set up publishers
        self.q_matrix_pub = rospy.Publisher("/q_learning/q_matrix", QMatrix, queue_size = 10)
        self.robot_action_pub = rospy.Publisher("/q_learning/robot_action", RobotMoveDBToBlock, queue_size = 10)

        # Set up subscriber
        rospy.Subscriber("q_learning/reward", QLearningReward, self.reward_received)

        # Sleep for 1 second to ensure everything is set up
        rospy.sleep(1)

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

        # Initialize and publish Q-matrix
        self.q_matrix = QMatrix()
        self.initialize_q_matrix()
        self.q_matrix_pub.publish(self.q_matrix)

        # Initialize current state and keep track of the next state
        self.curr_state = 0
        self.next_state = -1

        # Initialize action index
        self.action = -1

        self.initialized = True


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

        if not self.initialized:
            return
        
        curr_state = self.curr_state
        actions_in_row = self.action_matrix[curr_state]
        filtered_actions_in_row = filter(lambda x: x != -1, actions_in_row)
        
        if len(filtered_actions_in_row) == 0:
            return

        selected_action = choice(filtered_actions_in_row)
        self.action = selected_action
        self.next_state = actions_in_row.index(selected_action)
        
        db = self.actions[selected_action]["dumbbell"]
        block = self.actions[selected_action]["block"]

        robot_action = RobotMoveDBToBlock()
        robot_action.robot_db = db
        robot_action.block_id = block
        self.robot_action_pub.publish(robot_action)


    def update_q_matrix(self, data):
        """ Apply the Q-learning algorithm to update and publish the Q-matrix """

        curr_state = self.curr_state
        next_state = self.next_state
        action = self.action

        alpha = 1
        gamma = 0.5

        q_st_at = self.q_matrix.q_matrix[curr_state].q_matrix_row[action]
        q_st_at = q_st_at + alpha * (data.reward + gamma * max(self.q_matrix.q_matrix[next_state].q_matrix_row) - q_st_at)
        self.q_matrix.q_matrix[curr_state].q_matrix_row[action] = q_st_at

        self.curr_state = self.next_state

        self.q_matrix_pub.publish(self.q_matrix)


    def is_converged(self):
        # TODO: Check if the Q-matrix converged
        return


    def save_q_matrix(self):
        """ Save Q-matrix as a csv file once it's converged to avoid retraining """
        data = self.q_matrix.q_matrix
        data = np.asarray(data)
        np.savetxt("./q_matrix.csv", data, delimiter = ',')

    
    def reward_received(self, data):
        # TODO: Process reward after an action, includes update and check convergence,
        #  data argument is QLearningReward
        return


if __name__ == "__main__":
    node = QLearning()
