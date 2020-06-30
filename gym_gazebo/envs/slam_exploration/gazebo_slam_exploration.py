import gym
import numpy as np
import os
import rospy
import roslaunch
import subprocess
import time
import math

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding
from std_srvs.srv import Empty

import actionlib
from simple_movement.msg import FlyToAction, FlyToGoal
from geometry_msgs.msg import PoseStamped, Pose


class GazeboSlamExplorationEnv(gazebo_env.GazeboEnv):

    def _pause(self, msg):
        programPause = raw_input(str(msg))

    def position_callback(self, data):
        self.position = data.pose

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboSlamExploration-v0.launch")

        self.action_space = spaces.Discrete(26)
        # self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

        self.initial_latitude = None
        self.initial_longitude = None

        self.current_latitude = None
        self.current_longitude = None

        self.diff_latitude = None
        self.diff_longitude = None

        self.max_distance = 0

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Defining client for Fly to position
        self.client = actionlib.SimpleActionClient('fly_to', FlyToAction)
        self.client.wait_for_server()

        self.position = Pose()
        # Subscribing to the position of the drone
        rospy.Subscriber("/position_drone", PoseStamped,
                         self.position_callback)

        # countdown = 3
        # while countdown > 0:
        #     print("Taking off in in %ds" % countdown)
        #     countdown -= 1
        #     time.sleep(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _state(self, action):
        return discretized_ranges, done

    def step(self, action):
        # The robot can move in a 3d environment so the robot is placed in a cube 3x3 that indicates the directions where it can go
        # Define 26 action one for each sub box of the big box
        # F = Forward
        # B = Backwards
        # U = Up
        # D = Down
        # L = Left
        # R = Right
        # Combinations of thes basic direction can be used. Example: FUR -> Means the upper, right corner going forward of the 3x3 box
        # Starting from the Up and going in anticlock versus
        # Front Line

        goal = FlyToGoal()

        goal.pose = PoseStamped()
        goal.distance_converged = 0.3
        goal.yaw_converged = 1
        
        if action == 0:  # FORWARD 
            print("Going Forward")
            goal.pose.pose.position.x = self.position.position.x+1
        elif action == 1:  # FU
            print("Going FU")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
        elif action == 2:  # FUL
            print("Going FUL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1
        elif action == 3:  # FL
            print("Going FL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 4:  # FDL
            print("Going FDL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 5:  # FD
            print("Going FD")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 6:  # FDR
            print("Going FDR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 7:  # FR
            print("Going FR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 8:  # FUR
            print("Going FUR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Central line
        elif action == 9:  # U
            print("Going U")
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 10:  # UL
            print("Going UL")
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 11:  # L
            print("Going L")
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 12:  # DL
            print("Going DL")
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 13:  # D
            print("Going D")
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 14:  # DR
            print("Going DR")
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 15:  # R
            print("Going R")
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 16:  # UR
            print("Going UR")
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Back line
        elif action == 17:  # B
            print("Going B")
            goal.pose.pose.position.x = self.position.position.x-1

        elif action == 18:  # BU
            print("Going BU")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 19:  # BUL
            print("Going BUL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 20:  # BL
            print("Going BL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 21:  # BDL
            print("Going BDL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 22:  # BD
            print("Going BD")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 23:  # BDR
            print("Going BDR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 24:  # BR
            print("Going BR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 25:  # BUR
            print("Going BUR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1

        # Send /set_position message and wait till the point is not reached
        self.client.send_goal(goal)
        self.client.wait_for_result()

        observation = self._get_state()
        done = self.position.position.x >= 10
        reward = 0

        if done:
            reward = 100
        else:
            reward = self.position.position.x -1
        return observation, reward, done, {}

    def _killall(self, process_name):
        pids = subprocess.check_output(["pidof", process_name]).split()
        for pid in pids:
            os.system("kill -9 "+str(pid))

    def _get_state(self):  # Get position and map
        # read position data
        data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/mavros/global_position/global', NavSatFix, timeout=5)
        #     except:
        #         pass

        return self.position

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')

        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_world service call failed")

        # Restart octomap

        self.initial_latitude = None
        self.initial_longitude = None
        

        return self._get_state()
