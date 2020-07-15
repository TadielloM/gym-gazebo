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
# from octomap_msgs.msg import Octomap
from octomap_msgs.msg import Octomap
import octomap
from libraries import conversions
import ros_numpy

def print_dbg(data):
    pass
    #print(data)

class GazeboSlamExplorationEnv(gazebo_env.GazeboEnv):
    CUBE_SIZE = 1
    def _pause(self, msg):
        programPause = raw_input(str(msg))

    def position_callback(self, data):
        self.position = data.pose

    # def readTree(self, octree, msg):
    #     # std::stringstream datastream;
    #     if (len(msg.data) > 0):
    #         # datastream.write((const char*) &msg.data[0], msg.data.size());
    #         octree.readBinaryData(msg)
    

    def map_callback(self, data):
        print len(data.data)
        self.map = conversions.msgToMap(data)
        # print type(self.map)

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
        return self.position, self.map

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        rospy.wait_for_service("/octomap_server/reset")

        octomap_resetted = False
        try:
            if(self.reset_octomap()):
                octomap_resetted = True
                print("CAZZO 1")
            
            
            
            # while 
        except rospy.ServiceException as exc:
            print("Service reset octomap did not process request: " + str(exc))

        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_world service call failed")

        
        # Restart octomap

        self.initial_latitude = None
        self.initial_longitude = None
        
        return self._get_state(), octomap_resetted


    def collisionLine(self):
        #Setting the upper right corner and bottom left corner of the box 
        upper_right_corner = [self.position.position.x+(CUBE_SIZE+(CUBE_SIZE/2)), self.position.position.y+(CUBE_SIZE+(CUBE_SIZE/2)), self.position.position.z+(CUBE_SIZE+(CUBE_SIZE/2))]
        bottom_left_corner = [self.position.position.x-(CUBE_SIZE+(CUBE_SIZE/2)), self.position.position.y-(CUBE_SIZE+(CUBE_SIZE/2)), self.position.position.z-(CUBE_SIZE+(CUBE_SIZE/2))]



#To see collisions
# bool STLAEPlanner::collisionLine(std::shared_ptr<point_rtree> stl_rtree, Eigen::Vector4d p1, Eigen::Vector4d p2,
#                                 double r)
# {
# octomap::point3d start(p1[0], p1[1], p1[2]);
# octomap::point3d end(p2[0], p2[1], p2[2]);

# point bbx_min(std::min(p1[0], p2[0]) - r, std::min(p1[1], p2[1]) - r, std::min(p1[2], p2[2]) - r);
# point bbx_max(std::max(p1[0], p2[0]) + r, std::max(p1[1], p2[1]) + r, std::max(p1[2], p2[2]) + r);

# box query_box(bbx_min, bbx_max);
# std::vector<point> hits;
# stl_rtree->query(boost::geometry::index::intersects(query_box), std::back_inserter(hits));

# double lsq = (end - start).norm_sq();
# double rsq = r * r;

# for (size_t i = 0; i < hits.size(); ++i)
# {
#     octomap::point3d pt(hits[i].get<0>(), hits[i].get<1>(), hits[i].get<2>());

#     if (CylTest_CapsFirst(start, end, lsq, rsq, pt) > 0 or (end - pt).norm() < r)
#     {
#     return true;
#     }
# }

# return false;
# }

# //-----------------------------------------------------------------------------
# // Name: CylTest_CapsFirst
# // Orig: Greg James - gjames@NVIDIA.com
# // Lisc: Free code - no warranty & no money back.  Use it all you want
# // Desc:
# //    This function tests if the 3D point 'pt' lies within an arbitrarily
# // oriented cylinder.  The cylinder is defined by an axis from 'pt1' to 'pt2',
# // the axis having a length squared of 'lsq' (pre-compute for each cylinder
# // to avoid repeated work!), and radius squared of 'rsq'.
# //    The function tests against the end caps first, which is cheap -> only
# // a single dot product to test against the parallel cylinder caps.  If the
# // point is within these, more work is done to find the distance of the point
# // from the cylinder axis.
# //    Fancy Math (TM) makes the whole test possible with only two dot-products
# // a subtract, and two multiplies.  For clarity, the 2nd mult is kept as a
# // divide.  It might be faster to change this to a mult by also passing in
# // 1/lengthsq and using that instead.
# //    Elminiate the first 3 subtracts by specifying the cylinder as a base
# // point on one end cap and a vector to the other end cap (pass in {dx,dy,dz}
# // instead of 'pt2' ).
# //
# //    The dot product is constant along a plane perpendicular to a vector.
# //    The magnitude of the cross product divided by one vector length is
# // constant along a cylinder surface defined by the other vector as axis.
# //
# // Return:  -1.0 if point is outside the cylinder
# // Return:  distance squared from cylinder axis if point is inside.
# //
# //-----------------------------------------------------------------------------
# float STLAEPlanner::CylTest_CapsFirst(const octomap::point3d& pt1, const octomap::point3d& pt2, float lsq, float rsq,
#                                     const octomap::point3d& pt)
# {
# float dx, dy, dz;     // vector d  from line segment point 1 to point 2
# float pdx, pdy, pdz;  // vector pd from point 1 to test point
# float dot, dsq;

# dx = pt2.x() - pt1.x();  // translate so pt1 is origin.  Make vector from
# dy = pt2.y() - pt1.y();  // pt1 to pt2.  Need for this is easily eliminated
# dz = pt2.z() - pt1.z();

# pdx = pt.x() - pt1.x();  // vector from pt1 to test point.
# pdy = pt.y() - pt1.y();
# pdz = pt.z() - pt1.z();

# // Dot the d and pd vectors to see if point lies behind the
# // cylinder cap at pt1.x, pt1.y, pt1.z

# dot = pdx * dx + pdy * dy + pdz * dz;

# // If dot is less than zero the point is behind the pt1 cap.
# // If greater than the cylinder axis line segment length squared
# // then the point is outside the other end cap at pt2.

# if (dot < 0.0f || dot > lsq)
#     return (-1.0f);
# else
# {
#     // Point lies within the parallel caps, so find
#     // distance squared from point to line, using the fact that sin^2 + cos^2 = 1
#     // the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
#     // Carefull: '*' means mult for scalars and dotproduct for vectors
#     // In short, where dist is pt distance to cyl axis:
#     // dist = sin( pd to d ) * |pd|
#     // distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
#     // dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
#     // dsq = pd * pd - dot * dot / lengthsq
#     //  where lengthsq is d*d or |d|^2 that is passed into this function

#     // distance squared to the cylinder axis:

#     dsq = (pdx * pdx + pdy * pdy + pdz * pdz) - dot * dot / lsq;

#     if (dsq > rsq)
#     return (-1.0f);
#     else
#     return (dsq);  // return distance squared to axis
# }
# }