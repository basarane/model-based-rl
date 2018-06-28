nvidia-docker run -it --name $1 --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$HOME/.Xauthority:/root/.Xauthority:rw" -e PASSWORD=123 -v /home/ersin:/host ersin/tensorflow_torch:1.4.0_10 /bin/bash

#nvidia-docker run -it --name $1 -p $(($2 + 8890)):8888 -p $(($2 + 6008)):6006 -p $(($2 + 6002)):6000 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$HOME/.Xauthority:/root/.Xauthority:rw" -e PASSWORD=123 -v /home/ersin:/host ersin/tensorflow_torch:1.4.0_7 /bin/bash


#nvidia-docker run -it --name $1 -p 8889:8889 -p 6007:6007 --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$HOME/.Xauthority:/root/.Xauthority:rw" -e PASSWORD=123 -v /home/ersin:/host ersin/tensorflow_torch:1.4.0_7 /bin/bash
#sudo nvidia-docker run -it -p 8888:8888 -p 6006:6006 -e PASSWORD=123 -v /home/ubuntu:/host gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
