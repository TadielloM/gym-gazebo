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


class GazeboSlamExplorationEnv(gazebo_env.GazeboEnv):

    def _pause(self, msg):
        programPause = raw_input(str(msg))

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboSlamExploration-v0.launch")

        print("Arrivo qua")


        self.action_space = spaces.Discrete(26)
        #self.observation_space = spaces.Box(low=0, high=20) #laser values
        self.reward_range = (-np.inf, np.inf)

        self.initial_latitude = None
        self.initial_longitude = None

        self.current_latitude = None
        self.current_longitude = None

        self.diff_latitude = None
        self.diff_longitude = None

        self.max_distance = 1.6

        
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)


        countdown = 10
        while countdown > 0:
            print ("Taking off in in %ds"%countdown)
            countdown-=1
            time.sleep(1)

        print("Arrivo qua 2")        
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
        print("Arrivo qua 3")
        #Front Line
        if action == 0: #FORWARD
            print("Going Forward")
        elif action == 1: #FU
            print("Going FU")
        elif action == 2: #FUL
            print("Going FUL")
        elif action == 3: #FL
            print("Going FL")
        elif action == 4: #FDL
            print("Going FU")
        elif action == 5: #FD
            print("Going FD")
        elif action == 6: #FDR
            print("Going FDR")
        elif action == 7: #FR
            print("Going FR")
        elif action == 8: #FUR
            print("Going FUR")
        #Central line
        
        elif action == 9: #U
            print("Going U")
        elif action == 10: #UL
            print("Going UL")
        elif action == 11: #L
            print("Going L")
        elif action == 12: #DL
            print("Going DL")
        elif action == 13: #D
            print("Going D")
        elif action == 14: #DR
            print("Going DR")
        elif action == 15: #R
            print("Going R")
        elif action == 16: #UR
            print("Going UR")
        
        #Back line 
        elif action == 17: #B
            print("Going B")
        elif action == 18: #BU
            print("Going BU")
        elif action == 19: #BUL
            print("Going BUL")
        elif action == 20: #BL
            print("Going BL")
        elif action == 21: #BDL
            print("Going BDL")
        elif action == 22: #BD
            print("Going BD")
        elif action == 23: #BDR
            print("Going BDR")
        elif action == 24: #BR
            print("Going BR")
        elif action == 25: #BUR
            print("Going BUR")
        
        # Send /set_position message and wait till the point is not reached

        print("Arrivo qua 4")
        observation = self._get_state()
        print("Arrivo qua 5")
        dist = self.center_distance()
        done = dist > self.max_distance
        print("Arrivo qua 6")
        reward = 0
        if done:
            reward = -100
        else:
            reward = 10 - dist * 8
        print("Arrivo qua 7")
        return observation, reward, done, {}


    def _killall(self, process_name):
        pids = subprocess.check_output(["pidof",process_name]).split()
        for pid in pids:
            os.system("kill -9 "+str(pid))

    def _get_state(self): # Get position and map 
        #read position data
        data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/mavros/global_position/global', NavSatFix, timeout=5)
        #     except:
        #         pass

        self.current_latitude = 0
        self.current_longitude = 0

        if self.initial_latitude == None and self.initial_longitude == None:
            self.initial_latitude = self.current_latitude
            self.initial_longitude = self.current_longitude
            print("Initial latitude : %f, Initial Longitude : %f" % (self.initial_latitude,self.initial_longitude,))

        print("Current latitude : %f, Current Longitude : %f" % (self.current_latitude,self.current_longitude,))

        self.diff_latitude = self.current_latitude - self.initial_latitude
        self.diff_longitude = self.current_longitude - self.initial_longitude

        print("Diff latitude: %f, Diff Longitude: %f" % (self.diff_latitude,self.diff_longitude,))

        return self.diff_latitude, self.diff_longitude

    def center_distance(self):
        return math.sqrt(self.diff_latitude**2 + self.diff_longitude**2)

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_world service call failed")

        # Restart octomap

        self.initial_latitude = None
        self.initial_longitude = None

        return self._get_state()
