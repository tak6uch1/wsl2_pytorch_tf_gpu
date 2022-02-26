#!/bin/bash

# First, download .deb file from https://developer.nvidia.com/rdp/cudnn-download
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.3.2.44_1.0-1_amd64.deb

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo apt-get -y update

apt-cache search cudnn
sudo apt-get -y update
sudo apt-get -y install libcudnn8 libcudnn8-dev
dpkg -l | grep cudnn

