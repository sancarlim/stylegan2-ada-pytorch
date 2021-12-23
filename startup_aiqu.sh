#!/bin/sh

apt-get update  # To get the latest package lists

pip install  --use-feature=2020-resolver click requests tqdm pyspng efficientnet_pytorch seaborn ninja torchtoolbox imageio-ffmpeg==0.4.3 
apt-get install ffmpeg libsm6 libxext6 imagemagick openssh-server -y

git config --global user.email "sandra.carrascol@uah.es"
git config --global user.name "sandracl72"

apt-get install byobu -y
byobu-enable
byobu-enable-prompt
. ~/.bashrc

mkdir /run/sshd
PATH=/usr/sbin/:$PATH
useradd -m sandra
passwd sandra
chown -R sandra /workspace/stylegan2-ada-pytorch
# /usr/sbin/sshd -D -p <port>