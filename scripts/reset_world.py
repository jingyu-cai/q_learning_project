#!/usr/bin/env python3

import rospy


from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from q_learning_project.msg import QLearningReward
from std_msgs.msg import Header

from random import shuffle
from tf.transformations import quaternion_from_euler, euler_from_quaternion


print_header = "-" * 10

class ResetWorld(object):

    def __init__(self):

        print(print_header + "RESETWORLD" + print_header)

        # initialize this node
        rospy.init_node('reset_world_q_learning')

        # reward amounts
        self.positive_reward_amount = 100
        self.negative_reward_amount = 0

        self.robot_dbs = ["red", "green", "blue"]

        # db model names
        self.db_model_names = {
            "red": "robot_dumbbell_red",
            "green": "robot_dumbbell_green",
            "blue": "robot_dumbbell_blue"
        }

        # reset positions and orientations of dbs
        self.reset_xyz_positions_of_dbs = [
            Point(x=1.0635, y=-0.5, z=0.1905),
            Point(x=1.0635, y=0.0, z=0.1905),
            Point(x=1.0635, y=0.5, z=0.1905)
        ]
        reset_quat_orientation_of_dbs_list = quaternion_from_euler(1.5708, 0.0, 0.0)
        self.reset_quat_orientation_of_dbs = Quaternion()
        self.reset_quat_orientation_of_dbs.x = reset_quat_orientation_of_dbs_list[0]
        self.reset_quat_orientation_of_dbs.y = reset_quat_orientation_of_dbs_list[1]
        self.reset_quat_orientation_of_dbs.z = reset_quat_orientation_of_dbs_list[2]
        self.reset_quat_orientation_of_dbs.w = reset_quat_orientation_of_dbs_list[3]

        # numbered block model names
        self.numbered_block_model_names = {
            1: "number1",
            2: "number2",
            3: "number3"
        }

        # reset position and orientations of the numbered blocks
        self.reset_numbered_blocks_positions = [
            Point(x=-2.4, y=-2.0, z=0.4),
            Point(x=-2.4, y=0.0, z=0.4),
            Point(x=-2.4, y=2.0, z=0.4)
        ]
        self.reset_quat_orientation_of_numbered_blocks = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # reset position and orientation of the robot
        self.robot_model_name = "robot"
        self.robot_reset_position = Point(x=0.0, y=0.0, z=0.0)
        self.robot_reset_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


        # goal locations
        self.goal_matches_dbs_to_numbered_blocks = {
            "red": 3,
            "green": 1,
            "blue": 2
        }

        # current location of the numbered blocks
        self.current_numbered_blocks_locations = None

        # current locations of the robot dbs relative to the blocks
        self.current_robot_db_locations = {
            "red": 0,
            "green": 0,
            "blue": 0
        }

        # flag to keep track of the state of when we're resetting the world and when we're not
        # to avoid sending too many duplicate messages
        self.reset_world_in_progress = False

        # keep track of the iteration number
        self.iteration_num = 0
        
        # ROS subscribers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_received)

        # ROS publishers
        self.model_states_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.reward_pub = rospy.Publisher("/q_learning/reward", QLearningReward, queue_size=10)

        self.run()


    def is_in_front_of_any_block(self, pose):

        block_id_in_front_of = 0

        for block_id in range(1, 4):
            if (self.is_in_front_of_block_id(pose, block_id)):
                block_id_in_front_of = block_id

        return block_id_in_front_of



    def is_in_front_of_block_id(self, pose, block_id):

        x = pose.position.x
        y = pose.position.y

        x_delta_range = 0.5

        y_delta_range = 0.4

        if (block_id not in self.current_numbered_blocks_locations):
            print("is_in_front_of_1_block(): invalid block_id")
            return False

        x_min = self.current_numbered_blocks_locations[block_id].x + 0.4
        x_max = x_min + x_delta_range
        y_min = self.current_numbered_blocks_locations[block_id].y - y_delta_range
        y_max = self.current_numbered_blocks_locations[block_id].y + y_delta_range

        if (x >= x_min and x <= x_max and y >= y_min and y <= y_max):
            return True
        else:
            return False


    def model_states_received(self, data):

        # get the initial locations of the three numbered blocks
        if (self.current_numbered_blocks_locations == None):
            self.current_numbered_blocks_locations = {}
            for block_id in self.numbered_block_model_names:
                block_idx = data.name.index(self.numbered_block_model_names[block_id])
                self.current_numbered_blocks_locations[block_id] = data.pose[block_idx].position

        db_block_mapping = {}

        for robot_db in self.robot_dbs:
            db_idx = data.name.index(self.db_model_names[robot_db])
            db_block_mapping[robot_db] = self.is_in_front_of_any_block(data.pose[db_idx])

        # if a dumbbell has moved in front of a numbered block, send a reward
        is_robot_db_update = False
        for robot_db in self.robot_dbs:
            if (self.current_robot_db_locations[robot_db] != db_block_mapping[robot_db]):
                is_robot_db_update = True
                # print(robot_db, " from ", self.current_robot_db_locations[robot_db], " to ", db_block_mapping[robot_db])
                self.current_robot_db_locations[robot_db] = db_block_mapping[robot_db]

        if (is_robot_db_update and not self.reset_world_in_progress):

            # assign reward
            reward_amount = -1
            reached_goal_locations = True
            for robot_db in self.robot_dbs:
                if (self.goal_matches_dbs_to_numbered_blocks[robot_db] != db_block_mapping[robot_db]):
                    reached_goal_locations = False

            if (reached_goal_locations):
                reward_amount = self.positive_reward_amount
            else:
                reward_amount = self.negative_reward_amount

            # publish reward
            reward_msg = QLearningReward()
            reward_msg.header = Header(stamp=rospy.Time.now())
            reward_msg.reward = reward_amount
            reward_msg.iteration_num = self.iteration_num
            self.reward_pub.publish(reward_msg)
            print("Published reward: ", reward_amount)




        # if all 3 dumbbells are in front of numbered block, reset the world
        dbs_in_final_position = True
        for robot_db in self.robot_dbs:
            if (db_block_mapping[robot_db] == 0):
                dbs_in_final_position = False

        if (dbs_in_final_position and not self.reset_world_in_progress):

            self.reset_world_in_progress = True

            # reset world (dbs position)
            shuffle(self.robot_dbs)
            for i in range(len(self.robot_dbs)):
                p = Pose(position=self.reset_xyz_positions_of_dbs[i],
                         orientation=self.reset_quat_orientation_of_dbs)
                t = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
                m_name = self.db_model_names[self.robot_dbs[i]]

                robot_db_model_state = ModelState(model_name=m_name, pose=p, twist=t)

                self.model_states_pub.publish(robot_db_model_state)

                self.current_robot_db_locations[self.robot_dbs[i]] = 0

            # reset world (numbered blocks positions)
            numbered_blocks_position_order = [1, 2, 3]
            shuffle(numbered_blocks_position_order)
            for i in range(len(numbered_blocks_position_order)):
                p = Pose(position=self.reset_numbered_blocks_positions[i],
                         orientation=self.reset_quat_orientation_of_numbered_blocks)
                t = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
                m_name = self.numbered_block_model_names[numbered_blocks_position_order[i]]

                numbered_block_model_state = ModelState(model_name=m_name, pose=p, twist=t)

                self.model_states_pub.publish(numbered_block_model_state)

                self.current_numbered_blocks_locations[numbered_blocks_position_order[i]] = self.reset_numbered_blocks_positions[i]

            # reset world (robot position)
            p = Pose(position=self.robot_reset_position, orientation=self.robot_reset_orientation)
            t = Twist(linear=Vector3(0,0,0), angular=Vector3(0,0,0))
            robot_model_state = ModelState(model_name=self.robot_model_name, pose=p, twist=t)
            self.model_states_pub.publish(robot_model_state)


        elif (not dbs_in_final_position and self.reset_world_in_progress):
            self.reset_world_in_progress = False
            self.iteration_num += 1




    def run(self):
        rospy.spin()





if __name__=="__main__":

    node = ResetWorld()