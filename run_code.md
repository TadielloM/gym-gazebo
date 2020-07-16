To run the docker use:
```rocker --x11 --nvidia --pulse --git --oyr-run-arg "--mount type=bind,source=/home/matteo/git-repos/gym-gazebo,target=/root/gym-gazebo" gym_gazebo:latest```
Change the mount source with the folder where you have your code

Then go to ```~/gym-gazebo``` and run ```pip install -e .```
Then go to ```~/gym-gazebo/gym_gazebo/envs/installation/``` and run ```bash drone_velodyne_setup.bash```

For simplicity a file is created to do that run ```bash ~/gym-gazebo/initialize_env.sh``` to initialize the docker container

Then to run the code run ```python ~/gym-gazebo/examples/slam_exploration/slam_exploration_qlearn.py```
