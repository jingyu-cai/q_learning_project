# q_learning_project

## Implementation Plan

### Names of team members
Jingyu Cai [jcai23@uchicago.edu](mailto:jcai23@uchicago.edu)

Zhou Xing [zhouxing@uchicago.edu](mailto:zhouxing@uchicago.edu)

### Testing Steps

#### Q Learning

	
	$roscore
	$roslaunch q_learning_project training.launch
	$rosrun q_learning_project phantom_robot_movement.py

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
