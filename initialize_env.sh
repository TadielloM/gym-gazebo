#! /bin/sh
set -e 

cd ~/gym-gazebo
pip3 install -e .

if [ -z "$GAZEBO_MODEL_PATH" ]; then
echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/gym-gazebo/gym_gazebo/envs/assets/models" >> ~/.bashrc
else
bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=/root/gym-gazebo/gym_gazebo/envs/assets/models'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_MINE" ]; then
echo "export GYM_GAZEBO_WORLD_MINE=/root/gym-gazebo/gym_gazebo/envs/assets/worlds/mine.world" >> ~/.bashrc
else
bash -c 'sed "s,GYM_GAZEBO_WORLD_MINE=[^;]*,'GYM_GAZEBO_WORLD_MINE=/root/gym-gazebo/gym_gazebo/envs/assets/worlds/mine.world'," -i ~/.bashrc'
fi

if [ -z "$GAZEBO_PLUGIN_PATH" ]; then
echo "export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/ros_ws/devel/lib/" >> ~/.bashrc
else
bash -c 'sed "s,GAZEBO_PLUGIN_PATH=[^;]*,'GAZEBO_PLUGIN_PATH=/root/ros_ws/devel/lib/'," -i ~/.bashrc'
fi

# cd ~/gym-gazebo/examples/slam_exploration/


# cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && swig -python -c++ conversions.i
# cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -c -Wall -fpic conversions_wrap.cxx -I/usr/include/python2.7 -I/opt/ros/melodic/include/
# cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -shared conversions_wrap.o -L/opt/ros/melodic/lib -loctomap -lm -o _conversions.so

# exec "$@"