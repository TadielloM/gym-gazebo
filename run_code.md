# HOW TO RUN THIS CODE

First install docker and rocker
For docker follow the instruction on the docker website.
For rocker 
```pip3 install rocker```
and the extension for rocker
```pip3 install git+https://github.com/sloretz/off-your-rocker.git```
To run the docker use:
```shell
rocker --x11 --nvidia --pulse --git --oyr-run-arg "--mount type=bind,source=/home/matteo/git-repos/gym-gazebo,target=/root/gym-gazebo" gym-gazebo:latest
```
Change the mount source with the folder where you have your code
For tensorboard run 
```shell
rocker --x11 --nvidia --pulse --git --oyr-run-arg "--mount type=bind,source=/home/matteo/git-repos/gym-gazebo,target=/root/gym-gazebo -p 0.0.0.0:6006:6006" gym-gazebo:latest
```
since tensorboard run on port 6006, i link the local host port 6006 with the port of the docker with ```-p 0.0.0.0:6006:6006```

Then go to ```~/gym-gazebo``` and run ```pip install -e .```
Then go to ```~/gym-gazebo/gym_gazebo/envs/installation/``` and run ```bash drone_velodyne_setup.bash```

For simplicity a file is created to do that run ```bash ~/gym-gazebo/initialize_env.sh``` to initialize the docker container (done automatically as ENTRYPOINT of the container)


Then to run the code run ```python ~/gym-gazebo/examples/slam_exploration/slam_exploration_qlearn.py```

for tensorboard run
```tensorboard --logdir /root/ray_results/ --host 0.0.0.0 --port 6006```
tensorboard --logdir /home/tadistyle/saved_models_and_log/logs --port 6006   #in gcp
