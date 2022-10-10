FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y
RUN apt install -y \
wget \
unzip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /FEAT
RUN pip install tensorflow wandb tensorboardX scipy pandas json2html cockpit-for-pytorch
RUN pip install --upgrade numpy
RUN apt install -y tmux
