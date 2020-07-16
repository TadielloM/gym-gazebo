#! /bin/bash
cd ~/gym-gazebo
pip install -e .
cd ~/gym-gazebo/gym_gazebo/envs/installation/
bash setup_melodic.bash 
bash drone_velodyne_setup.bash
cd ~/gym-gazebo/examples/slam_exploration/

cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && swig -python -c++ conversions.i
cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -c -Wall -fpic conversions_wrap.cxx -I/usr/include/python2.7 -I/opt/ros/melodic/include/
cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -shared conversions_wrap.o -L/opt/ros/melodic/lib -loctomap -lm -o _conversions.so