# q_learning_project

## Implementation Plan

### Names of team members
Jingyu Cai [jcai23@uchicago.edu](mailto:jcai23@uchicago.edu)

Zhou Xing [zhouxing@uchicago.edu](mailto:zhouxing@uchicago.edu)

### Testing Steps

#### Q Learning

	$ roscore
	$ roslaunch q_learning_project training.launch
	$ rosrun q_learning_project phantom_robot_movement.py


#### Robot Perception & Movement

	$ roscore
	$ roslaunch q_learning_project turtlebot3_intro_robo_manipulation.launch
    $ roslaunch q_learning_project action.launch


<span style="color:red">**IMPORTANT:**</span> remember clicking the <span style="color:yellow">**PLAY**</span> button!	

### Component implementation and testing
- **Q-learning algorithm**
    - **Executing the Q-learning algorithm**
   
     Follow the Q-learning algorithm below, and test it by comparing some manual calculation results with printed results from code.
        <img src="./qlearning_algo.png" width=800>

    - **Determining when the Q-matrix has converged**
    
      Set a threshold value (e.g. `epsilon = 0.01`) so that if the sum of variation of all entries in the matrix is less than `epsilon`, we would denote this matrix as "unchanged" from the previous state. If such "static" status keeps for a certain number of loop (e.g. 5 loops), we would determine this matrix as converged. We would test it by printing out the matrix once it reaches the static status, and manually check it's consistent with the optimal trajectory.

    - **Once the Q-matrix has converged, how to determine which actions the robot should take to maximize expected reward**

      For each state, take the action with the highest Q-value. Similar to taking a greedy strategy following the Q matrix. The testing would be the same as the one in the previous section: print out the converged matrix, and check if the strategy it indicates is identical to one of the optimal solutions.

- **Robot perception**
    - **Determining the identities and locations of the three colored dumbbells**
  
        The idea would be using **Sensory-Motor Control**, and following the *Line Follower* class exercise from Class 3: 
        
        For each color of the three dumbbells:
        1. Defining the range for pixels in this color in the image feed to determine the identities of the dumbbells
        2. After getting locations of all pixels in this color, compute the center of the "color block", and turn the head of robot to put it at the center of the camera.
        
        For testing, we would look at the GUI of Gazebo, and see if the robot would turn to the wanted dumbbell. 

    - **Determining the identities and locations of the three numbered blocks**
  
        We would navigate to the front of these three numbered blocks (so that the number would be at the center of the camera), and use `keras_ocr` for digit recognition by implementing the pre-trained models into our script. After determining their identities, we would also determine the locations of the blocks using the robot's odometry and LiDAR data. For testing, once a number is recognized, we would print out this number and check if it's consistent with the one shown on Gazebo.

- **Robot manipulation & movement**
    - **Picking up and putting down the dumbbells with the OpenMANIPULATOR arm**

        Following the *Forward & Inverse Kinematics* section in class 8, we would calculate the desired angle for each joint, based on the size of the dumbbells. The putting down process is expected to be the reverse of the picking up process. For testing, it would also simply be a "visual check" on Gazebo to see if the robot can lift and put down the dumbbells.

    - **Navigating to the appropriate locations to pick up and put down the dumbbells**
  
        We would implement proportional control to enable the robot to navigate to the dumbbell of a particular color. It would be similar to the *Person Follower* in warm-up project. For testing, we would also look at Gazebo to check if the robot moves to the wanted dumbbell.

### Timeline (tentative)
- Robot perception: May 3rd
- Robot manipulation and movement: May 7th
- Q-learning: May 10th
- The rest of the time would be used for tuning and optimization.
- DDL: Wednesday, May 12 11:00am CST

## Writeup

### Objectives description
For this project, the objective is to first train a Q-matrix based on the Q-learning algorithm. Then, with the trained Q-matrix that specifices what action to take in a particular state to maximize the received reward, we need to make the robot perform a sequence of perceptions and movements to place each dumbbell in front of the correct block.

### High-level description
To determine which dumbbell belongs in front of which block, we used reinforcement learning by employing the Q-learning algorithm. The robot can be in different states (defined by where the dumbbells are) and different actions that it can take in each state (defined by the movement of a dumbbell to a block), and each action in a given state results in a reward. Specifically, we used the pre-defined `actions.txt`, `states.txt`, and `action_matrix.txt` to define a set of functions that finds the best action to take in a given state that maximizes the received reward, resulting in a Q-matrix. Once the Q-matrix is trained, we will be able to choose an action from each state and apply that onto the movement of the robot in Gazebo.

### Q-learning algorithm description
- **Selecting and executing actions for the robot (or phantom robot) to take**: For this component, we first initialized an empty Q-matrix populated with 0s and published that to the `QMatrix()` message, and set the current state to 0. Then, for each given state, we chose a random and valid action for the robot to take (move a dumbbell to a block) from the pre-defined files that elicit possible actions and states in the `action_states` folder, and published that action as a `RobotMoveDBToBlock()` message to the robot/phantom robot. Lastly, after processing the received reward, we would jump into the next state and repeat the above process. The relevant code are located in `q_learning.py`.
	- `init()`: In here, we set up the necessary variables for selecting and executing actions for the robot/phantom robot, including the publishers for `QMatrix()` and `RobotMoveDBToBlock()`, the action matrix, actions, and states from the pre-defined files, and variables to keep track of the robot's state. We also initialized the Q-matrix and published it, and chose a random action to begin.
	- `initialize_q_matrix()`: In here, we assigned every cell within the Q-matrix with a value of 0 to start.
	- `select_random_action()`: In here, we identified the valid actions to take given a current state from `self.action_matrix`, and randomly selected one of those actions using numpy's `choice()` function. Then, after updating what the next state would be for the selected action, we published the dumbbell color and block number via a `RobotMoveDBToBlock()` message for the robot/phantom robot to execute.
