#! /bin/bash
cd ~/gym-gazebo
pip install -e .
cd ~/gym-gazebo/gym_gazebo/envs/installation/
bash drone_velodyne_setup.bash
cd ~/gym-gazebo/examples/slam_exploration/
