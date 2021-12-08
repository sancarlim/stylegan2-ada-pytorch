#!/bin/sh

apt-get update  # To get the latest package lists

pip install click requests tqdm pyspng efficientnet_pytorch seaborn ninja torchtoolbox imageio-ffmpeg==0.4.3 
apt-get install ffmpeg libsm6 libxext6 imagemagick -y
apt-get install byobu -y
byobu-enable
byobu-enable-prompt
. ~/.bashrc
