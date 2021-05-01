#!/usr/bin/env python3

import rospy


from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from q_learning_project.msg import QLearningReward
from q_learning_project.msg import RobotMoveDBToBlock

from std_msgs.msg import Header

from random import shuffle
from tf.transformations import quaternion_from_euler, euler_from_quaternion

print_header = "-" * 10

class ResetWorld(object):

    def __init__(self):

        # initialize this node
        rospy.init_node('virtual_reset_world_q_learning')

        print("==**********=*====resetworld.py=========*=*====")

        # reward amounts
        self.positive_reward = 100
        self.negative_reward = 0

        # goal locations
        self.goal_robot_db = {
            "red": 3,
            "green": 1,
            "blue": 2
        }

        # current locations of the robot dbs relative to the blocks
        self.current_robot_db = {
            "red": 0,
            "green": 0,
            "blue": 0
        }

        # keep track of the iteration number
        self.iteration_num = 0

        # ROS subscribe to the topic publishing actions for the robot to take
        rospy.Subscriber("/q_learning/robot_action", RobotMoveDBToBlock, self.send_reward)

        # ROS publishers
        self.reward_pub = rospy.Publisher("/q_learning/reward", QLearningReward, queue_size=10)

        self.run()

    def send_reward(self, data):
        print(data)
        # update locations based on latest db movement
        self.current_robot_db[data.robot_db] = data.block_id

        # default reward
        reward_amount = self.positive_reward
        reset_world = True

        # check if dbs are in correct position & if the world should be reset
        for robot_db in self.current_robot_db.keys():
            if self.current_robot_db[robot_db] != self.goal_robot_db[robot_db]:
                reward_amount = self.negative_reward


            if self.current_robot_db[robot_db] == 0:
                reset_world = False


        # prepare reward msg
        reward_msg = QLearningReward()
        reward_msg.header = Header(stamp=rospy.Time.now())
        reward_msg.reward = reward_amount
        reward_msg.iteration_num = self.iteration_num
        self.reward_pub.publish(reward_msg)
        print(print_header + "Published reward " + print_header, reward_amount)

        # increment iteration if world needs to be reset
        # reset db positions if world needs to be rest
        if reset_world:
            print("reseting the world")
            self.iteration_num += 1

            for robot_db in self.current_robot_db.keys():
                self.current_robot_db[robot_db] = 0




    def run(self):
        rospy.spin()

if __name__=="__main__":

    node = ResetWorld()
