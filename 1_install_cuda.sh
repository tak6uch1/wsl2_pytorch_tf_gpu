#!/bin/bash

sudo apt-get update
sudo apt-get -y upgrade

# Install CUDA 11.5
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-wsl-ubuntu-11-5-local_11.5.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-5-local_11.5.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-5-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

echo 'export CUDA_PATH=/usr/local/cuda-11.5' >> ${HOME}/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:${LD_LIBRARY_PATH}' >> ${HOME}/.bashrc
echo 'export PATH=/usr/local/cuda-11.5/bin:${PATH}' >> ${HOME}/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ${HOME}/.bashrc
source ${HOME}/.bashrc

sudo ln -s /usr/local/cuda-11.5/lib64/libcusolver.so.11 /usr/local/cuda-11.5/lib64/libcusolver.so.10
cat /usr/local/cuda-11.5/version.json

# Install third party tool (optional)
sudo apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# Build sample program (optional)
sudo rm -rf cuda-samples > /dev/null 2>&1
sudo git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery/
sudo make
sudo ./deviceQuery
cd ../../../../cuda-samples/Samples/1_Utilities/bandwidthTest/
sudo make
sudo ./bandwidthTest
cd ../../../../cuda-samples/Samples/4_CUDA_Libraries/simpleCUBLAS
sudo make
sudo ./simpleCUBLAS
cd ../../../../cuda-samples/Samples/4_CUDA_Libraries/simpleCUFFT/
sudo make
sudo ./simpleCUFFT

