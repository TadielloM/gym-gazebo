#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_MINE" ]; then
bash -c 'echo "export GYM_GAZEBO_WORLD_MINE="`pwd`/../assets/worlds/mine.world >> ~/.bashrc'
else
bash -c 'sed "s,GYM_GAZEBO_WORLD_MINE=[^;]*,'GYM_GAZEBO_WORLD_MINE=`pwd`/../assets/worlds/mine.world'," -i ~/.bashrc'
fi

exec bash # reload bash