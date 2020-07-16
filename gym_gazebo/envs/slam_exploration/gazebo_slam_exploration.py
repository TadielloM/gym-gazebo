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
from simple_movement.msg import FlyToAction , FlyToGoal
from geometry_msgs.msg import PoseStamped, Pose
# from octomap_msgs.msg import Octomap
from octomap_msgs.msg import Octomap
import octomap
from libraries import conversions
import ros_numpy

def print_dbg(data):
    # pass
    print(data)

class GazeboSlamExplorationEnv(gazebo_env.GazeboEnv):
    CUBE_SIZE = 1
    MAX_TIME_EPISODE = 10 #60*10
    def _pause(self, msg):
        programPause = raw_input(str(msg))

    def position_callback(self, data):
        self.position = data.pose

    def map_callback(self, data):
        # print len(data.data)
        self.map = conversions.msgToMap(data)
        # print type(self.map)

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboSlamExploration-v0.launch")

        self.action_space = spaces.Discrete(26)
        # self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

        self.max_distance = 0
        self.episode_time = 0


        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Defining client for Fly to position
        self.client = actionlib.SimpleActionClient('fly_to', FlyToAction)
        self.client.wait_for_server()

        self.position = Pose()
        resolution = 0.1
        self.map = octomap.OcTree(resolution)
        # print type(self.map)
        # Subscribing to the position of the drone
        rospy.Subscriber("/position_drone", PoseStamped, self.position_callback)

        # Subscribing to the map
        rospy.Subscriber("/octomap_binary", Octomap , self.map_callback)

        

        self.reset_octomap = rospy.ServiceProxy("/octomap_server/reset", Empty)
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
            print_dbg("Going Forward")
            goal.pose.pose.position.x = self.position.position.x+1
        elif action == 1:  # FU
            print_dbg("Going FU")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
        elif action == 2:  # FUL
            print_dbg("Going FUL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1
        elif action == 3:  # FL
            print_dbg("Going FL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 4:  # FDL
            print_dbg("Going FDL")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 5:  # FD
            print_dbg("Going FD")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 6:  # FDR
            print_dbg("Going FDR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 7:  # FR
            print_dbg("Going FR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 8:  # FUR
            print_dbg("Going FUR")
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Central line
        elif action == 9:  # U
            print_dbg("Going U")
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 10:  # UL
            print_dbg("Going UL")
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 11:  # L
            print_dbg("Going L")
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 12:  # DL
            print_dbg("Going DL")
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 13:  # D
            print_dbg("Going D")
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 14:  # DR
            print_dbg("Going DR")
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 15:  # R
            print_dbg("Going R")
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 16:  # UR
            print_dbg("Going UR")
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Back line
        elif action == 17:  # B
            print_dbg("Going B")
            goal.pose.pose.position.x = self.position.position.x-1

        elif action == 18:  # BU
            print_dbg("Going BU")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 19:  # BUL
            print_dbg("Going BUL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 20:  # BL
            print_dbg("Going BL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 21:  # BDL
            print_dbg("Going BDL")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 22:  # BD
            print_dbg("Going BD")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 23:  # BDR
            print_dbg("Going BDR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 24:  # BR
            print_dbg("Going BR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 25:  # BUR
            print_dbg("Going BUR")
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1

        # Send /set_position message and wait till the point is not reached
        self.client.send_goal(goal)
        self.client.wait_for_result()

        observation = self._get_state()

        #Condition for end the episode: If episode time is higher then MAX_TIME_EPISODE
        print ( rospy.Duration(self.MAX_TIME_EPISODE) )
        print "Elapsed time: ", rospy.Time.now()-self.episode_time
        if rospy.Time.now()-self.episode_time > rospy.Duration(self.MAX_TIME_EPISODE): # or self.position.position.x >= 10 
            done = True
        else:
            done = False
        print 
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
        return self.position, self.map

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        rospy.wait_for_service("/octomap_server/reset")
        
        
        # Reset octomap
        try:
            print ("Resetting octomap")
            self.reset_octomap()
            print ("Octomap resetted")
        except rospy.ServiceException as exc:
            print("Service reset octomap did not process request: " + str(exc))

        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_world service call failed")
        
        self.episode_time = rospy.Time.now()

        return self._get_state()