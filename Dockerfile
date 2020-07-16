FROM osrf/ros:melodic-desktop-full

# WORKDIR ~

# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# RUN source ~/.bashrc

RUN apt update && \
    apt upgrade -y && \
    apt-get install -y \
    apt-utils \
    python-pip python3-vcstool python3-pyqt4 \
    pyqt5-dev-tools \
    libbluetooth-dev libspnav-dev \
    pyqt4-dev-tools libcwiid-dev \
    cmake gcc g++ qt4-qmake libqt4-dev \
    libusb-dev libftdi-dev \
    python3-defusedxml python3-vcstool \
    ros-melodic-octomap-msgs        \
    ros-melodic-joy                 \
    ros-melodic-geodesy             \
    ros-melodic-octomap-ros         \
    ros-melodic-control-toolbox     \
    ros-melodic-pluginlib	       \
    ros-melodic-trajectory-msgs     \
    ros-melodic-control-msgs	       \
    ros-melodic-std-srvs 	       \
    ros-melodic-nodelet	       \
    ros-melodic-urdf		       \
    ros-melodic-rviz		       \
    ros-melodic-kdl-conversions     \
    ros-melodic-eigen-conversions   \
    ros-melodic-tf2-sensor-msgs     \
    ros-melodic-pcl-ros \
    ros-melodic-navigation \
    ros-melodic-sophus \
    ros-melodic-ros-numpy \
    libdynamicedt3d-dev \ 
    gfortran \
    python-skimage \
    psmisc \
    python-tk \
    htop \
    swig

RUN pip install --upgrade gym h5py tensorflow-gpu keras pandas liveplot pydot pyparsing scikit-image==0.14.2 octomap-python

#For Deep Q learning
RUN cd ~ && git clone git://github.com/Theano/Theano.git
RUN cd ~/Theano/ && python setup.py develop

RUN mkdir -p ~/ros_ws/src && cd ~/ros_ws/src && \
    git clone https://github.com/OctoMap/octomap_mapping.git \
    && git clone https://bitbucket.org/DataspeedInc/velodyne_simulator.git \
    && git clone https://TadielloM:Mass21effect@github.com/Moes96/servo_pkg.git \
    && git clone --single-branch --branch movement_as_service https://github.com/TadielloM/simple_movement.git

RUN cd ~/ros_ws/ && /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make"
RUN echo "source /root/ros_ws/devel/setup.bash" >> /root/.bashrc
RUN echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> /root/.bashrc


VOLUME [ "/root/gym-gazebo" ]

#When needed for the installation
# RUN cd ~ && git clone --single-branch --branch develop https://github.com/TadielloM/gym-gazebo.git
# RUN pip install -e ~/gym-gazebo
# RUN cd ~/gym-gazebo/gym_gazebo/envs/installation && bash setup_melodic.bash 
# RUN cd ~/gym-gazebo/gym_gazebo/envs/installation && bash drone_velodyne_setup.bash 

# RUN cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && swig -python -c++ conversions.i
# RUN cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -c -Wall -fpic conversions_wrap.cxx -I/usr/include/python2.7 -I/opt/ros/melodic/include/
# RUN cd /root/gym-gazebo/gym_gazebo/envs/slam_exploration/libraries && g++ -shared conversions_wrap.o -L/opt/ros/melodic/lib -loctomap -lm -o _conversions.so
