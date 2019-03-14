FROM ubuntu:16.04

# install software-properties-common
RUN apt-get update
RUN apt-get install -y software-properties-common

# install python3.5
RUN add-apt-repository -y ppa:fkrull/deadsnakes
RUN apt-get update
RUN apt-get install -y python3.5

# install git
RUN apt-get update
RUN apt-get install -y git-core

# install pip
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# install opencv
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender1

# install scipy
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy

# install tensorflow
RUN apt-get update
RUN apt-get install -y python3-dev
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.12.0

# install vim
RUN apt-get install -y vim





