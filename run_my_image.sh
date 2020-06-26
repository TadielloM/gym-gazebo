# XAUTH=/tmp/.docker.xauth
# if [ ! -f $XAUTH ]
# then
#     xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
#     if [ ! -z "$xauth_list" ]
#     then
#         echo $xauth_list | xauth -f $XAUTH nmerge -
#     else
#         touch $XAUTH
#     fi
#     chmod a+r $XAUTH
# fi

# sudo docker run -it --rm \
#     --user=$(id -u $USER):$(id -g $USER) \
#     --env="DISPLAY" \
#     --volume="/etc/group:/etc/group:ro" \
#     --volume="/etc/passwd:/etc/passwd:ro" \
#     --volume="/etc/shadow:/etc/shadow:ro" \
#     --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --runtime=nvidia \
#     gym-gazebo:latest \
#     bash
xhost +local:root

docker run --gpus all -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    gym-gazebo:latest \
    bash

xhost -local:root


#with rocker i run:
docker run -it   --rm \
    -v /home/matteo/.gitconfig:/root/.gitconfig:ro \
    --mount type=bind,source=/home/matteo/git-repos/gym-gazebo,target=/root/gym-gazebo \
    -v /run/user/1000/pulse:/run/user/1000/pulse \
    --device /dev/snd \
    -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
    -v /run/user/1000/pulse/native:/run/user/1000/pulse/native \ 
    --group-add 29 \
    -e DISPLAY -e TERM   -e QT_X11_NO_MITSHM=1   -e XAUTHORITY=/tmp/.docker.xauth 
    -v /tmp/.docker.xauth:/tmp/.docker.xauth \   
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/localtime:/etc/localtime:ro 8d7f5b9a4620 bash