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
# import octomap
# from libraries import conversions
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# import pcl_ros.point_cloud as pcl
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import collections

def print_dbg(data):
    print(data)
    # pass

def print_dbg2(data):
    print(data)
    # pass
    

AGENT_HISTORY_LENGTH = 4
RESIZED_X = 200
RESIZED_Y = 200
RESIZED_Z = 200
RESIZED_DATA = 2 #is voxel full is voxel empty
MAX_NUM_POINTS = 100000

class TestGazeboSlamExplorationEnv(gazebo_env.GazeboEnv):
    CUBE_SIZE = 1
    MAX_TIME_EPISODE = 120 #60*10

    def _pause(self, msg):
        programPause = raw_input(str(msg))

    def position_callback(self, data):
        self.position = data.pose

            # print("x : ",p[0] ," y: " , p[1] ," z: ",p[2])
        # print('The length of the point cloud is: ', len(self.point_cloud))

    def __init__(self,config):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TestGazeboSlamExploration-v0.launch")
        self.espisode_num = 0
        self.action_space = spaces.Discrete(26)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(RESIZED_X, RESIZED_Y, RESIZED_Z, RESIZED_DATA), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(3), dtype=np.float)
        self.reward_range = (-np.inf, np.inf)

        self.episode_time = 0
        self.prev_map_size = 0
        self.point_cloud = None
        self.num_actions = 0
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Defining client for Fly to position
        self.client = actionlib.SimpleActionClient('fly_to', FlyToAction)
        self.client.wait_for_server()

        self.position = Pose()
        # Subscribing to the position of the drone
        rospy.Subscriber("/position_drone", PoseStamped, self.position_callback)        
        countdown = 5
        while countdown > 0:
            print("Taking off in %ds" % countdown)
            countdown -= 1
            time.sleep(1)
    #     self._seed()
        

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _state(self, action):
        return discretized_ranges, done

    def step(self, action ): #num_thread
        
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
        goal.pose.header.frame_id='map'
        goal.distance_converged = 0.2
        goal.yaw_converged = 0.2
        goal.pose.pose.position.x = self.position.position.x
        goal.pose.pose.position.y = self.position.position.y
        goal.pose.pose.position.z = self.position.position.z

        goal.pose.pose.orientation.x = 0 
        goal.pose.pose.orientation.y = 0
        goal.pose.pose.orientation.z = 0
        goal.pose.pose.orientation.w = 1
        self.num_actions +=1
        if action == 0:  # FORWARD 
            print_dbg((self.num_actions," Going Forward"))
            goal.pose.pose.position.x = self.position.position.x+1
        elif action == 1:  # FU
            print_dbg((self.num_actions," Going FU"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
        elif action == 2:  # FUL
            print_dbg((self.num_actions," Going FUL"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1
        elif action == 3:  # FL
            print_dbg((self.num_actions," Going FL"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 4:  # FDL
            print_dbg((self.num_actions," Going FDL"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 5:  # FD
            print_dbg((self.num_actions," Going FD"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 6:  # FDR
            print_dbg((self.num_actions," Going FDR"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 7:  # FR
            print_dbg((self.num_actions," Going FR"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 8:  # FUR
            print_dbg((self.num_actions," Going FUR"))
            goal.pose.pose.position.x = self.position.position.x+1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Central line
        elif action == 9:  # U
            print_dbg((self.num_actions," Going U"))
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 10:  # UL
            print_dbg((self.num_actions," Going UL"))
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 11:  # L
            print_dbg((self.num_actions," Going L"))
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 12:  # DL
            print_dbg((self.num_actions," Going DL"))
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 13:  # D
            print_dbg((self.num_actions," Going D"))
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 14:  # DR
            print_dbg((self.num_actions," Going DR"))
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 15:  # R
            print_dbg((self.num_actions," Going R"))
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 16:  # UR
            print_dbg((self.num_actions," Going UR"))
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1


        # Back line
        elif action == 17:  # B
            print_dbg((self.num_actions," Going B"))
            goal.pose.pose.position.x = self.position.position.x-1

        elif action == 18:  # BU
            print_dbg((self.num_actions," Going BU"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1

        elif action == 19:  # BUL
            print_dbg((self.num_actions," Going BUL"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 20:  # BL
            print_dbg((self.num_actions," Going BL"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 21:  # BDL
            print_dbg((self.num_actions," Going BDL"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y-1

        elif action == 22:  # BD
            print_dbg((self.num_actions," Going BD"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1

        elif action == 23:  # BDR
            print_dbg((self.num_actions," Going BDR"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 24:  # BR
            print_dbg((self.num_actions," Going BR"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.y = self.position.position.y+1

        elif action == 25:  # BUR
            print_dbg((self.num_actions," Going BUR"))
            goal.pose.pose.position.x = self.position.position.x-1
            goal.pose.pose.position.z = self.position.position.z+1
            goal.pose.pose.position.y = self.position.position.y+1

        

        # Send /set_position message and wait till the point is not reached
        # print ( "num: ", num_thread, "goal:" )
        # print ( "x: ",goal.pose.pose.position.x )
        # print ( "y: ",goal.pose.pose.position.y )  
        # print ( "z: ",goal.pose.pose.position.z ) 
        self.client.send_goal(goal)
        self.client.wait_for_result()

        # print (num_thread, "result received ", action)
        observation = self._get_state()
        
        #Condition for end the episode: If episode time is higher then MAX_TIME_EPISODE
        # print ( rospy.Duration(self.MAX_TIME_EPISODE) )
        # print "Elapsed time: ", rospy.Time.now()-self.episode_time
        if rospy.Time.now()-self.episode_time > rospy.Duration(self.MAX_TIME_EPISODE): # or self.position.position.x >= 10 
            done = True
            print_dbg2("DONE= true episode completed")
        else:
            done = False
        
        if done:
            reward = 0
        else:
            reward = 1

        print("The reward is: ", reward )
        return observation, reward, done, {}

    def _killall(self, process_name):
        pids = subprocess.check_output(["pidof", process_name]).split()
        for pid in pids:
            os.system("kill -9 "+str(pid))

    def _get_state(self):  # Get position and map
        return self.position

    def reset(self):
        print_dbg2("Resetting the environment")
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        self.prev_map_size = 0
        self.episode_time = 0
        print_dbg2("Ended episode: "+ str(self.espisode_num))
        self.espisode_num += 1
        
        try:
            print("resetting proxy")
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as exc:
            print("/gazebo/reset_world service call failed" + str(exc))


        self.episode_time = rospy.Time.now()

        return self._get_state()




